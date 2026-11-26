// Shared kernel body. p selects one convolution and grid_z selects its output chunk and K split.
#if AJI_GCONV_ARCH_OK
    constexpr int TH = WARPS, CP = KCH + 8, SP = COUTP + 8, NT = COUTP / 8, NJ = NT / 2, KCK = KCH / 16;
    [[maybe_unused]] constexpr int VPP = KCH / 8;
    constexpr int TR = S == 1 ? TH + 2 : 2 * TH + 1, TC = S == 1 ? TW + 2 : 2 * TW + 1;
    static_assert(KCH % 16 == 0 && COUTP % 16 == 0, "channel counts are multiples of 16");
    static_assert(IN == 0 || KCH == 16, "NCHW inputs are one 16-channel chunk");
    extern __shared__ __align__(16) unsigned char smraw[];
    half *sm = reinterpret_cast<half *>(smraw);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int by = blockIdx.y;
    if constexpr (BAT) {
        const int b = by / p.tiles_y;
        by -= b * p.tiles_y;
        const size_t ip = (size_t)b * p.IH * p.IW, op = (size_t)b * p.H * p.W;
        p.src += ip * p.sstride;
        p.dst += op * p.dstride;
        if (p.r1) p.r1 += op * p.r1s;
        if (p.r2) p.r2 += op * p.r2s;
        if (p.ws) p.ws += op * p.coutp_all;
    }
    const int tx0 = blockIdx.x * TW, ty0 = by * TH;
    const int iy0 = S * ty0 - 1, ix0 = S * tx0 - 1;
    const int och = grid_z / p.ksplit, kz = grid_z % p.ksplit;
    const uint4 *wt = p.w + och * p.wchunk;
    const int units = p.nchunks * 9, u0 = kz * units / p.ksplit, u1 = (kz + 1) * units / p.ksplit;

    uint32_t acc[NT][2];
    [[maybe_unused]] float accf[NT][4];
#pragma unroll
    for (int n = 0; n < NT; n++) {
        acc[n][0] = acc[n][1] = 0u;
#pragma unroll
        for (int k = 0; k < 4; k++) accf[n][k] = 0.f;
    }

#pragma unroll 1
    for (int c = u0 / 9; c < p.nchunks && c * 9 < u1; c++) {
        if constexpr (IN == 0) {
            for (int i = tid; i < TR * TC * VPP; i += WARPS * 32) {
                const int px = i / VPP, v = i % VPP, r = px / TC, col = px % TC;
                const int gc = S == 1 ? col : (col < TW + 1 ? 2 * col : 2 * (col - (TW + 1)) + 1);
                const int yy = iy0 + r, xx = ix0 + gc;
                const bool ok = yy >= 0 && yy < p.IH && xx >= 0 && xx < p.IW;
                const size_t sp = p.up2 ? (size_t)(yy >> 1) * (p.IW >> 1) + (xx >> 1) : (size_t)yy * p.IW + xx;
                g_cp16(sm + px * CP + v * 8, ok ? p.src + sp * p.sstride + c * KCH + v * 8 : p.src, ok ? 16 : 0);
            }
            g_cp_wait();
        } else {
            for (int px = tid; px < TR * TC; px += WARPS * 32) {
                const int r = px / TC, col = px % TC;
                const int gc = S == 1 ? col : (col < TW + 1 ? 2 * col : 2 * (col - (TW + 1)) + 1);
                const int yy = iy0 + r, xx = ix0 + gc;
                __align__(16) half v[16];
#pragma unroll
                for (int k = 0; k < 16; k++) v[k] = __float2half(0.f);
                if (yy >= 0 && yy < p.IH && xx >= 0 && xx < p.IW) {
#pragma unroll
                    for (int k = 0; k < 3; k++) v[k] = p.src[(size_t)k * p.sstride + (size_t)yy * p.IW + xx];
                }
                reinterpret_cast<uint4 *>(sm + px * CP)[0] = reinterpret_cast<uint4 *>(v)[0];
                reinterpret_cast<uint4 *>(sm + px * CP)[1] = reinterpret_cast<uint4 *>(v)[1];
            }
        }
        __syncthreads();
        const int t0 = max(u0 - c * 9, 0), t1 = min(u1 - c * 9, 9);
#pragma unroll 1
        for (int t = t0; t < t1; t++) {
            const int dy = t / 3, dx = t % 3, px = lane & 15;
            const int row = S == 1 ? warp + dy : 2 * warp + dy;
            const int col = S == 1 ? px + dx : ((dx & 1) ? TW + 1 + px : px + (dx >> 1));
            const half *ab = sm + (row * TC + col) * CP + (lane >> 4) * 8;
#pragma unroll
            for (int kc = 0; kc < KCK; kc++) {
                const int kk = c * KCK + kc;
                uint4 b[NJ];
#pragma unroll
                for (int j = 0; j < NJ; j++) b[j] = __ldg(wt + ((t * p.kct + kk) * NJ + j) * 32 + lane);
                uint32_t a[4];
                g_ldsm_x4(a, ab + kc * 16);
#pragma unroll
                for (int j = 0; j < NJ; j++) {
                    if constexpr (ACC) {
                        g_mma32(accf[2 * j], a, b[j].x, b[j].y);
                        g_mma32(accf[2 * j + 1], a, b[j].z, b[j].w);
                    } else {
                        g_mma(acc[2 * j], a, b[j].x, b[j].y);
                        g_mma(acc[2 * j + 1], a, b[j].z, b[j].w);
                    }
                }
            }
        }
        __syncthreads();
    }

    // epilogue: lane holds pixels g, g+8 of row `warp`, channels n*8 + 2tq, +1
    const int g = lane >> 2, tq = lane & 3, oy = ty0 + warp;
    if (p.ksplit > 1 || p.outf) {
        // fp32 partial sums to the workspace; k_gconv_reduce adds the bias and runs the epilogue
#pragma unroll
        for (int n = 0; n < NT; n++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int ox = tx0 + g + h * 8;
                if (oy >= p.H || ox >= p.W) continue;
                const float2 f = ACC ? make_float2(accf[n][2 * h], accf[n][2 * h + 1]) : __half22float2(u2h(acc[n][h]));
                *reinterpret_cast<float2 *>(p.ws + ((size_t)kz * (BAT ? p.nb : 1) * p.H * p.W + (size_t)oy * p.W + ox) * p.coutp_all + och * COUTP + n * 8 +
                                            tq * 2) = f;
            }
        return;
    }
    const half2 *bias = p.bias + och * COUTP / 2;
    GParams q = p;
    if (q.r1) q.r1 += och * COUTP;
    if (q.r2) q.r2 += och * COUTP;
#pragma unroll
    for (int n = 0; n < NT; n++) {
        const half2 bb = __ldg(bias + n * 4 + tq);
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int ox = tx0 + g + h * 8;
            half2 v;
            if constexpr (ACC) {
                const float2 bf = __half22float2(bb);
                v = __floats2half2_rn(accf[n][2 * h] + bf.x, accf[n][2 * h + 1] + bf.y);
            } else {
                v = __hadd2(u2h(acc[n][h]), bb);
            }
            if (oy < p.H && ox < p.W) v = g_epi(q, v, oy, ox, n * 8 + tq * 2);
            *reinterpret_cast<half2 *>(sm + (warp * TW + g + h * 8) * SP + n * 8 + tq * 2) = v;
        }
    }
    __syncthreads();

    if constexpr (OUT == 0) {
        constexpr int VO = COUTP / 8, VPR = TW * VO;
        half *dst = p.dst + och * COUTP;
        for (int q2 = tid; q2 < TH * VPR; q2 += WARPS * 32) {
            const int r = q2 / VPR, w = q2 % VPR, px = w / VO, v = w % VO, yy = ty0 + r, xx = tx0 + px;
            if (yy >= p.H || xx >= p.W) continue;
            *reinterpret_cast<uint4 *>(dst + ((size_t)yy * p.W + xx) * p.dstride + v * 8) =
                *reinterpret_cast<const uint4 *>(sm + (r * TW + px) * SP + v * 8);
        }
        if constexpr (SUMS) {
            // the tile's channel sums of the stored (fp16) values, one atomic per channel per CTA
            for (int c = tid; c < COUTP; c += WARPS * 32) {
                float sum = 0.f;
                for (int r = 0; r < TH; r++)
                    if (ty0 + r < p.H)
                        for (int px = 0; px < TW && tx0 + px < p.W; px++) sum += __half2float(sm[(r * TW + px) * SP + c]);
                atomicAdd(p.sums + och * COUTP + c, sum);
            }
        }
    } else {
        // phase scatter: channel sx*CHP + c of all output channels -> dst[R*y + sy][R*x + sx][c]; this CTA
        // holds channels [och*COUTP, och*COUTP + COUTP) (CHP a multiple of 8, so no vector straddles sx)
        const int R = p.R, CHP = p.coutp_all / R, OW = R * p.W;
        constexpr int VO = COUTP / 8;
        for (int q2 = tid; q2 < TH * TW * VO; q2 += WARPS * 32) {
            const int v = q2 % VO, px = (q2 / VO) % TW, r = q2 / (VO * TW);
            const int yy = ty0 + r, xx = tx0 + px;
            if (yy >= p.H || xx >= p.W) continue;
            const int gc = och * COUTP + v * 8, sx = gc / CHP, ch = gc % CHP;
            *reinterpret_cast<uint4 *>(p.dst + ((size_t)(R * yy + p.sy) * OW + R * xx + sx) * p.dstride + ch) =
                *reinterpret_cast<const uint4 *>(sm + (r * TW + px) * SP + v * 8);
        }
    }
#else
    __trap();
#endif
