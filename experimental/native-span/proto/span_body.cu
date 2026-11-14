// Prototype: fused SPANF3 body (5 SPABV2 blocks, C=48) as 3 kernels per block.
//   k1: a1 = relu(conv3x3(x) + b1)                      (block 5 also stores pre-act mid)
//   k2: a2 = relu(conv3x3(a1) + b2)
//   k3: o3 = conv3x3(a2) + b3; out = (x + o3) * tanh(Watt.o3 + batt)   (1x1 att fused)
// NHWC fp16 activations, WMMA m16n16k16 tensor cores, accumulator type templated.
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>

using namespace nvcuda;
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

constexpr int C = 48;          // feature channels
constexpr int TW = 16;         // output tile width  (one WMMA M=16 row of pixels)
constexpr int TH = 16;         // output tile height
constexpr int WARPS = 8;       // each warp: 2 output rows x 16 px x 48 ch
constexpr int HALO_PX = (TH + 2) * (TW + 2);

enum Mode { RELU = 0, RELU_AND_MID = 1, ATT = 2 };

struct BlockW {
    const half *w1, *w2, *w3, *watt;   // conv: [ky][kx][cin][cout]; att: [cin][cout]
    const float *b1, *b2, *b3, *batt;
};

__device__ __forceinline__ float fast_tanh(float x) {
    float y; asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
}

template <typename ACC>
__host__ __device__ constexpr int stage_bytes_per_warp(int mode) {
    // acc staging (32 px x 48) + for ATT a half copy of o3 for the 1x1 GEMM
    return 32 * C * (int)sizeof(ACC) + (mode == ATT ? 32 * C * 2 : 0);
}

template <typename ACC, int MODE>
__global__ void __launch_bounds__(WARPS * 32)
conv3x3_fused(const half* __restrict__ in, const half* __restrict__ resid, half* __restrict__ out,
              half* __restrict__ mid, const half* __restrict__ w, const float* __restrict__ b,
              const half* __restrict__ watt, const float* __restrict__ batt, int H, int W)
{
    extern __shared__ __align__(128) unsigned char smem[];
    half* tile = reinterpret_cast<half*>(smem);                    // [(TH+2)*(TW+2)][C]
    const int tx0 = blockIdx.x * TW, ty0 = blockIdx.y * TH;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

    // ---- load halo tile (zero padded), 16-byte vectors ----
    constexpr int VPP = C / 8;  // vectors per pixel
    for (int i = tid; i < HALO_PX * VPP; i += WARPS * 32) {
        int p = i / VPP, v = i % VPP;
        int yy = ty0 - 1 + p / (TW + 2), xx = tx0 - 1 + p % (TW + 2);
        uint4 val = make_uint4(0, 0, 0, 0);
        if (yy >= 0 && yy < H && xx >= 0 && xx < W)
            val = __ldg(reinterpret_cast<const uint4*>(in + ((size_t)yy * W + xx) * C) + v);
        reinterpret_cast<uint4*>(tile + p * C)[v] = val;
    }
    __syncthreads();

    // ---- implicit GEMM: per warp M=32 (2 rows), N=48, K=9*48 ----
    wmma::fragment<wmma::accumulator, 16, 16, 16, ACC> acc[2][3];
#pragma unroll
    for (int r = 0; r < 2; r++)
#pragma unroll
        for (int n = 0; n < 3; n++) wmma::fill_fragment(acc[r][n], (ACC)0);

    const int row0 = warp * 2;
#pragma unroll 1
    for (int t = 0; t < 9; t++) {
        const int dy = t / 3, dx = t % 3;
#pragma unroll
        for (int kc = 0; kc < 3; kc++) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[3];
#pragma unroll
            for (int n = 0; n < 3; n++)
                wmma::load_matrix_sync(bf[n], w + (t * C + kc * 16) * C + n * 16, C);
#pragma unroll
            for (int r = 0; r < 2; r++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
                wmma::load_matrix_sync(af, tile + ((row0 + r + dy) * (TW + 2) + dx) * C + kc * 16, C);
#pragma unroll
                for (int n = 0; n < 3; n++) wmma::mma_sync(acc[r][n], af, bf[n], acc[r][n]);
            }
        }
    }
    __syncthreads();  // all warps done reading the halo tile; reuse smem for staging

    ACC* st = reinterpret_cast<ACC*>(smem + warp * stage_bytes_per_warp<ACC>(MODE));  // [32 px][C]
#pragma unroll
    for (int r = 0; r < 2; r++)
#pragma unroll
        for (int n = 0; n < 3; n++)
            wmma::store_matrix_sync(st + (r * 16) * C + n * 16, acc[r][n], C, wmma::mem_row_major);
    __syncwarp();

    // Epilogue mapping: the warp's 32 px x 48 ch = 192 16-byte chunks; chunk q -> pixel q/6, vector q%6.
    // Consecutive lanes write consecutive 16 B in global memory (each 16-px row is 1536 contiguous bytes).
    constexpr int CHUNKS = 32 * (C / 8);

    if constexpr (MODE != ATT) {
#pragma unroll
        for (int q = lane; q < CHUNKS; q += 32) {
            const int px = q / (C / 8), v = q % (C / 8);
            const int oy = ty0 + row0 + (px >> 4), ox = tx0 + (px & 15);
            if (oy >= H || ox >= W) continue;
            const ACC* sp = st + px * C + v * 8;
            __align__(16) half o[8], m[8];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                float x = (float)sp[j] + __ldg(b + v * 8 + j);
                m[j] = __float2half(x);
                o[j] = __float2half(fmaxf(x, 0.f));
            }
            const size_t g = ((size_t)oy * W + ox) * C;
            reinterpret_cast<uint4*>(out + g)[v] = *reinterpret_cast<uint4*>(o);
            if constexpr (MODE == RELU_AND_MID) reinterpret_cast<uint4*>(mid + g)[v] = *reinterpret_cast<uint4*>(m);
        }
    } else {
        // o3 = acc + b3 -> half copy (A operand of the 1x1 att GEMM)
        half* o3 = reinterpret_cast<half*>(st + 32 * C);
#pragma unroll
        for (int q = lane; q < CHUNKS; q += 32) {
            const int px = q / (C / 8), v = q % (C / 8);
            const ACC* sp = st + px * C + v * 8;
            __align__(16) half t[8];
#pragma unroll
            for (int j = 0; j < 8; j++) t[j] = __float2half((float)sp[j] + __ldg(b + v * 8 + j));
            *reinterpret_cast<uint4*>(o3 + px * C + v * 8) = *reinterpret_cast<uint4*>(t);
        }
        __syncwarp();
        wmma::fragment<wmma::accumulator, 16, 16, 16, ACC> aacc[2][3];
#pragma unroll
        for (int r = 0; r < 2; r++)
#pragma unroll
            for (int n = 0; n < 3; n++) wmma::fill_fragment(aacc[r][n], (ACC)0);
#pragma unroll
        for (int kc = 0; kc < 3; kc++) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[3];
#pragma unroll
            for (int n = 0; n < 3; n++) wmma::load_matrix_sync(bf[n], watt + (kc * 16) * C + n * 16, C);
#pragma unroll
            for (int r = 0; r < 2; r++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
                wmma::load_matrix_sync(af, o3 + (r * 16) * C + kc * 16, C);
#pragma unroll
                for (int n = 0; n < 3; n++) wmma::mma_sync(aacc[r][n], af, bf[n], aacc[r][n]);
            }
        }
#pragma unroll
        for (int r = 0; r < 2; r++)
#pragma unroll
            for (int n = 0; n < 3; n++)
                wmma::store_matrix_sync(st + (r * 16) * C + n * 16, aacc[r][n], C, wmma::mem_row_major);
        __syncwarp();
#pragma unroll
        for (int q = lane; q < CHUNKS; q += 32) {
            const int px = q / (C / 8), v = q % (C / 8);
            const int oy = ty0 + row0 + (px >> 4), ox = tx0 + (px & 15);
            if (oy >= H || ox >= W) continue;
            const size_t g = ((size_t)oy * W + ox) * C;
            uint4 xr = __ldg(reinterpret_cast<const uint4*>(resid + g) + v);
            uint4 o3r = *reinterpret_cast<const uint4*>(o3 + px * C + v * 8);
            const half* xh = reinterpret_cast<const half*>(&xr);
            const half* oh = reinterpret_cast<const half*>(&o3r);
            const ACC* sp = st + px * C + v * 8;
            __align__(16) half o[8];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                float gate = fast_tanh((float)sp[j] + __ldg(batt + v * 8 + j));
                o[j] = __float2half((__half2float(xh[j]) + __half2float(oh[j])) * gate);
            }
            reinterpret_cast<uint4*>(out + g)[v] = *reinterpret_cast<uint4*>(o);
        }
    }
}

template <typename ACC>
struct Body {
    int H, W; size_t smem_conv, smem_att;
    void setup() {
        smem_conv = std::max((size_t)HALO_PX * C * 2, (size_t)WARPS * stage_bytes_per_warp<ACC>(RELU));
        smem_att = std::max((size_t)HALO_PX * C * 2, (size_t)WARPS * stage_bytes_per_warp<ACC>(ATT));
        CK(cudaFuncSetAttribute(conv3x3_fused<ACC, RELU>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_conv));
        CK(cudaFuncSetAttribute(conv3x3_fused<ACC, RELU_AND_MID>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_conv));
        CK(cudaFuncSetAttribute(conv3x3_fused<ACC, ATT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_att));
    }
    // x[0] = input, x[1..5] = block outputs
    void run(const BlockW* bw, half* const* x, half* a1, half* a2, half* mid, cudaStream_t s) {
        dim3 grid((W + TW - 1) / TW, (H + TH - 1) / TH), blk(WARPS * 32);
        for (int i = 0; i < 5; i++) {
            if (i == 4) conv3x3_fused<ACC, RELU_AND_MID><<<grid, blk, smem_conv, s>>>(x[i], nullptr, a1, mid, bw[i].w1, bw[i].b1, nullptr, nullptr, H, W);
            else        conv3x3_fused<ACC, RELU><<<grid, blk, smem_conv, s>>>(x[i], nullptr, a1, nullptr, bw[i].w1, bw[i].b1, nullptr, nullptr, H, W);
            conv3x3_fused<ACC, RELU><<<grid, blk, smem_conv, s>>>(a1, nullptr, a2, nullptr, bw[i].w2, bw[i].b2, nullptr, nullptr, H, W);
            conv3x3_fused<ACC, ATT><<<grid, blk, smem_att, s>>>(a2, x[i], x[i + 1], nullptr, bw[i].w3, bw[i].b3, bw[i].watt, bw[i].batt, H, W);
        }
    }
};

static std::vector<char> slurp(const char* p) {
    FILE* f = fopen(p, "rb"); if (!f) { printf("cannot open %s\n", p); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<char> v(n); fread(v.data(), 1, n, f); fclose(f); return v;
}

template <typename ACC>
static void go(const char* dir, const char* tag, int iters) {
    const int H = 540, W = 960; const size_t N = (size_t)H * W * C;
    std::string d(dir);
    auto wbuf = slurp((d + "/weights.bin").c_str());
    auto feat = slurp((d + "/feat.bin").c_str());
    char* dw; CK(cudaMalloc(&dw, wbuf.size())); CK(cudaMemcpy(dw, wbuf.data(), wbuf.size(), cudaMemcpyHostToDevice));
    BlockW bw[5]; size_t o = 0;
    auto take = [&](size_t bytes) { char* p = dw + o; o += bytes; return p; };
    for (int i = 0; i < 5; i++) {
        bw[i].w1 = (half*)take(9 * C * C * 2); bw[i].b1 = (float*)take(C * 4);
        bw[i].w2 = (half*)take(9 * C * C * 2); bw[i].b2 = (float*)take(C * 4);
        bw[i].w3 = (half*)take(9 * C * C * 2); bw[i].b3 = (float*)take(C * 4);
        bw[i].watt = (half*)take(C * C * 2);   bw[i].batt = (float*)take(C * 4);
    }
    if (o != wbuf.size()) { printf("weights size mismatch %zu vs %zu\n", o, wbuf.size()); exit(1); }
    half* x[6]; half *a1, *a2, *mid;
    for (auto& p : x) CK(cudaMalloc(&p, N * 2));
    CK(cudaMalloc(&a1, N * 2)); CK(cudaMalloc(&a2, N * 2)); CK(cudaMalloc(&mid, N * 2));
    CK(cudaMemcpy(x[0], feat.data(), N * 2, cudaMemcpyHostToDevice));
    Body<ACC> body{H, W}; body.setup();
    cudaStream_t s; CK(cudaStreamCreate(&s));
    body.run(bw, x, a1, a2, mid, s); CK(cudaStreamSynchronize(s)); CK(cudaGetLastError());
    std::vector<half> h(N);
    auto dump = [&](half* p, const char* name) {
        CK(cudaMemcpy(h.data(), p, N * 2, cudaMemcpyDeviceToHost));
        FILE* f = fopen((d + "/proto_" + tag + "_" + name + ".bin").c_str(), "wb"); fwrite(h.data(), 2, N, f); fclose(f);
    };
    dump(x[1], "outb1"); dump(x[5], "out5"); dump(mid, "mid5");
    // timing: whole body captured as one CUDA graph (as libaji / trtexec --useCudaGraph run it)
    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    body.run(bw, x, a1, a2, mid, s);
    CK(cudaStreamEndCapture(s, &g)); CK(cudaGraphInstantiate(&ge, g, 0));
    for (int i = 0; i < 200; i++) CK(cudaGraphLaunch(ge, s));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    std::vector<float> ms;
    for (int rep = 0; rep < 5; rep++) {
        CK(cudaEventRecord(e0, s));
        for (int i = 0; i < iters; i++) CK(cudaGraphLaunch(ge, s));
        CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1));
        float t; CK(cudaEventElapsedTime(&t, e0, e1)); ms.push_back(t / iters);
    }
    std::sort(ms.begin(), ms.end());
    printf("%s: body (5 blocks, 15 kernels) %.3f ms/frame median of 5x%d  [min %.3f max %.3f]  smem conv=%zu att=%zu\n",
           tag, ms[2], iters, ms[0], ms[4], body.smem_conv, body.smem_att);
}

int main(int argc, char** argv) {
    const char* dir = argc > 1 ? argv[1] : ".";
    int iters = argc > 2 ? atoi(argv[2]) : 500;
    const char* which = argc > 3 ? argv[3] : "both";
    if (strcmp(which, "32") != 0) go<half>(dir, "acc16", iters);
    if (strcmp(which, "16") != 0) go<float>(dir, "acc32", iters);
    return 0;
}
