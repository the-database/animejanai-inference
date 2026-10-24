/*
 * aji_encode — offline file->file upscale/interpolate CLI for libaji.
 *
 * aji_harness's libaji call sequence (create -> configure -> infer /
 * infer_rife -> drain -> destroy) wired to real libav demux/decode/encode/
 * mux. The missing piece that lets a non-player consumer (VideoJaNai) do
 * offline upscaling on top of libaji:
 *
 *   input file -> decode -> libaji (upscale + optional RIFE)
 *               -> [final resize] -> encode -> mux -> output file
 *
 * Single input -> single output; copies audio, subtitles, chapters and
 * attachments from the source.
 * Temporal (multi-frame) chains are driven through aji_ingest +
 * aji_infer_seq, each output inferred once its window has been ingested.
 *
 * Decode prefers NVDEC (frames stay GPU-resident, zero-copy into libaji and
 * out to NVENC, all sharing libav's CUDA context). --decoder cpu decodes to
 * host and uploads. See the companion spec for the full contract.
 *
 * --backend directml (Windows; the conf must say backend=DirectML too):
 * frames are shared D3D11 textures from libav D3D11 pools, decoded on the
 * CPU and uploaded, read back for encoding, 4:2:0 output only. No CUDA call
 * is made, and nvcuda/cudart are delay-loaded, so it runs on GPUs without
 * the NVIDIA driver.
 */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <libavcodec/avcodec.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavformat/avformat.h>
#include <libavutil/avstring.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#ifdef _WIN32
#include <libavutil/hwcontext_d3d11va.h>
#endif
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>

#include "aji.h"

/* ---- error-check macros -------------------------------------------------- */

#define CK(x) do { \
    cudaError_t e_ = (x); \
    if (e_ != cudaSuccess) { \
        loge("%s:%d: %s: %s", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
        goto fail; \
    } \
} while (0)

#define CU(x) do { \
    CUresult e_ = (x); \
    if (e_ != CUDA_SUCCESS) { \
        const char *s_ = NULL; cuGetErrorString(e_, &s_); \
        loge("%s:%d: %s: %s", __FILE__, __LINE__, #x, s_ ? s_ : "?"); \
        goto fail; \
    } \
} while (0)

#define AV(x) do { \
    int e_ = (x); \
    if (e_ < 0) { \
        char b_[AV_ERROR_MAX_STRING_SIZE]; \
        av_strerror(e_, b_, sizeof b_); \
        loge("%s:%d: %s: %s", __FILE__, __LINE__, #x, b_); \
        goto fail; \
    } \
} while (0)

/* ---- options ------------------------------------------------------------- */

enum decoder_mode { DEC_AUTO, DEC_NVDEC, DEC_CPU };
enum progress_mode { PROG_NONE, PROG_LINE, PROG_JSON };

/* output chroma subsampling + bit depth, chosen via --pix-fmt and decoupled
 * from the source. 16-bit 4:4:4 is libaji's intermediate, never the file. */
enum out_pixfmt { OUT_420P8, OUT_420P10, OUT_444P8, OUT_444P10 };

typedef struct {
    const char *input, *output;
    const char *conf, *model_dir, *rife_model_dir;
    const char *trtexec, *trtexec_env;
    int slot;
    const char *backend;
    const char *vcodec;
    const char *vquality;
    enum out_pixfmt pix_fmt;
    int final_h;            /* --final-resize-height, 0 = none */
    int final_pct;          /* --final-resize-factor, 0 = none */
    int overwrite;
    int no_audio, no_subs, no_chapters;
    enum progress_mode progress;
    int build_only;
    const char *log_path;
    enum decoder_mode decoder;
    int no_zerocopy;
    int pipeline_depth;

    /* direct mode (testing) */
    const char *engine;
    int max_w, max_h;
} options;

/* ---- logging: human/diagnostic -> stderr (or --log file); PROGRESS lines
 *      -> stdout, so stdout stays machine-parseable. ------------------------ */

static FILE *g_log;            /* stderr or the --log file */
static enum progress_mode g_progress = PROG_LINE;

static void loge(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vfprintf(g_log ? g_log : stderr, fmt, ap);
    va_end(ap);
    fputc('\n', g_log ? g_log : stderr);
}

static void aji_log_cb(void *opaque, int level, const char *msg)
{
    (void)opaque;
    if (level <= 2)
        loge("[aji:%d] %s", level, msg);
}

static void av_log_cb(void *p, int level, const char *fmt, va_list ap)
{
    (void)p;
    if (level > av_log_get_level())
        return;
    vfprintf(g_log ? g_log : stderr, fmt, ap);
}

static void progress(const char *phase, long long frame, long long total,
                     double fps, int pct)
{
    if (g_progress == PROG_NONE)
        return;
    if (g_progress == PROG_JSON)
        printf("{\"phase\":\"%s\",\"frame\":%lld,\"total\":%lld,"
               "\"fps\":%.2f,\"pct\":%d}\n", phase, frame, total, fps, pct);
    else
        printf("PROGRESS phase=%s frame=%lld total=%lld fps=%.2f pct=%d\n",
               phase, frame, total, fps, pct);
    fflush(stdout);
}

/* ---- colorimetry mapping (AVFrame/codecpar -> aji enums) ----------------- */

static int aji_matrix_from_av(enum AVColorSpace c)
{
    switch (c) {
    case AVCOL_SPC_BT470BG:
    case AVCOL_SPC_SMPTE170M:   return AJI_MATRIX_BT601;
    case AVCOL_SPC_BT2020_NCL:
    case AVCOL_SPC_BT2020_CL:   return AJI_MATRIX_BT2020;
    default:                    return AJI_MATRIX_BT709;
    }
}

static int aji_range_from_av(enum AVColorRange r)
{
    return r == AVCOL_RANGE_JPEG ? AJI_RANGE_FULL : AJI_RANGE_LIMITED;
}

static int aji_siting_from_av(enum AVChromaLocation l)
{
    switch (l) {
    case AVCHROMA_LOC_CENTER:   return AJI_SITING_CENTER;
    case AVCHROMA_LOC_TOPLEFT:  return AJI_SITING_TOPLEFT;
    default:                    return AJI_SITING_LEFT;
    }
}

/* ---- pipeline context ---------------------------------------------------- */

#define MAX_RING 32

typedef struct {
    AVFrame  *out;       /* aji output (or decoded, passthrough) CUDA frame */
    AVFrame  *in_ref;    /* decoded input kept alive across the async infer */
    uint64_t  ticket;
    int64_t   pts;       /* output PTS in the video stream time_base */
} slot;

typedef struct {
    options o;

    /* input */
    AVFormatContext *ifmt;
    int vstream;
    AVCodecContext  *dec;
    AVBufferRef     *hw_device;      /* CUDA hwdevice */
    CUcontext        cuctx;          /* shared context (from libav) */
    int              ctx_pushed;
    int              nvdec;          /* decoding on the GPU */
    int              dml;            /* DirectML backend: frames are shared D3D11
                                        textures (libav D3D11 frames), and no
                                        CUDA call is ever made */
    void            *d3d11_dev;      /* ID3D11Device* of the D3D11VA hwdevice */

    /* libaji */
    aji_ctx        *aji;
    cudaStream_t    stream;
    int             src_w, src_h;
    int             up_w, up_h;      /* libaji upscale output (pool dims) */
    int             out_w, out_h;    /* final encode dims (after resize) */
    int             ten_bit;         /* OUTPUT is 10-bit (from --pix-fmt) */
    int             is_444;          /* OUTPUT is 4:4:4 (from --pix-fmt) */
    int             src_aji_fmt;     /* input aji_frame.format from source */
    int             out_aji_fmt;     /* output aji_frame.format from --pix-fmt */
    enum AVPixelFormat out_pixfmt;   /* final software-encoder pix_fmt */
    int             passthrough;     /* no chain active: transcode only */
    int             rife;            /* RIFE active */
    int             rnum, rden;
    int             temporal_r;      /* temporal chain window radius, 0 = none */
    uint64_t        in_seq, out_seq; /* temporal: next frame to ingest / infer */
    int             rife_only;       /* RIFE with no scaling chain (aji_configure
                                        returned passthrough) */
    int             rife_first;      /* RIFE interpolates source frames, then
                                        each emitted frame is upscaled */
    int             pre_resize;      /* RIFE-first: downscale to work dims first */
    int             work_w, work_h;  /* RIFE-first input dims (= src unless
                                        pre_resize) */
    enum AVPixelFormat sw_fmt;       /* pool sw_format: NV12/P010LE/YUV444P16LE */
    AVBufferRef    *aji_pool;        /* CUDA frames pool aji writes into */
    AVBufferRef    *src_pool;        /* source-format CUDA frames at src dims
                                        (frames NVDEC did not deliver) */
    AVBufferRef    *work_pool;       /* source-format CUDA frames at work dims
                                        (RIFE-first downscales + interps) */
    struct SwsContext *in_sws;       /* host-side source repack */

    /* output */
    AVFormatContext *ofmt;
    AVCodecContext  *enc;
    int              vout;           /* output video stream index */
    int             *smap;           /* input stream idx -> output idx (-1) */
    AVRational       out_tb;         /* video stream time_base */
    AVRational       out_fps;
    int              opened;         /* header written */
    AVPacket       **aux_buf;        /* aux packets seen before header */
    int              aux_n, aux_cap;

    /* resize (nvenc + downscale): scale_cuda graph */
    AVFilterGraph   *graph;
    AVFilterContext *buf_src, *buf_sink;

    /* sw-encode scratch */
    struct SwsContext *sws;
    AVFrame *host_frame;             /* hwframe_transfer target */

    /* ring (non-RIFE pipelining) */
    slot ring[MAX_RING];
    int  ring_depth, ring_head, ring_count;

    /* RIFE state: two persistent upscaled CUDA frames + interp scratch */
    AVFrame *up_prev, *up_cur, *up_interp;
    int      have_prev;
    AVFrame *src_prev;               /* RIFE-first: previous work-res source */

    /* bookkeeping */
    int64_t  out_frames;
    int64_t  src_frames_est;
    int64_t  scene_dupes;
    int      color_space, color_range, color_pri, color_trc, chroma_loc;
} enc_ctx;

/* forward decls */
static int emit_output(enc_ctx *c, AVFrame *cuda_out);
static int drain_encoder(enc_ctx *c, int flush);

/* ---- decoder: force NVDEC's CUDA output format --------------------------- */

static enum AVPixelFormat g_hw_pixfmt = AV_PIX_FMT_CUDA;

static enum AVPixelFormat get_hw_format(AVCodecContext *ctx,
                                        const enum AVPixelFormat *fmts)
{
    const enc_ctx *c = ctx->opaque;
    for (const enum AVPixelFormat *p = fmts; *p != AV_PIX_FMT_NONE; p++)
        if (*p == g_hw_pixfmt)
            return *p;
    /* NVDEC can't take this stream (libav drops cuda from the list once its
     * hwaccel init fails, e.g. 4:4:4 H.264): --decoder auto carries on in
     * software, and to_device uploads the frames */
    if (c && c->o.decoder == DEC_AUTO) {
        for (const enum AVPixelFormat *p = fmts; *p != AV_PIX_FMT_NONE; p++) {
            const AVPixFmtDescriptor *d = av_pix_fmt_desc_get(*p);
            if (d && !(d->flags & AV_PIX_FMT_FLAG_HWACCEL)) {
                loge("NVDEC can't decode this stream; decoding on the CPU");
                return *p;
            }
        }
    }
    return AV_PIX_FMT_NONE;
}

/* ---- CUDA hwdevice + shared context -------------------------------------- */

static int setup_cuda(enc_ctx *c)
{
    if (av_hwdevice_ctx_create(&c->hw_device, AV_HWDEVICE_TYPE_CUDA,
                               NULL, NULL, 0) < 0) {
        loge("av_hwdevice_ctx_create(CUDA) failed");
        return -1;
    }
    AVHWDeviceContext   *dc = (AVHWDeviceContext *)c->hw_device->data;
    AVCUDADeviceContext *cu = (AVCUDADeviceContext *)dc->hwctx;
    c->cuctx = cu->cuda_ctx;

    /* Keep libav's driver context current for the whole run so the CUDA
     * runtime binds our stream/allocations to the same context libaji,
     * NVDEC and NVENC use. libav and libaji push/pop around their own ops. */
    if (cuCtxPushCurrent(c->cuctx) != CUDA_SUCCESS) {
        loge("cuCtxPushCurrent failed");
        return -1;
    }
    c->ctx_pushed = 1;
    if (cudaStreamCreate(&c->stream) != cudaSuccess) {
        loge("cudaStreamCreate failed");
        return -1;
    }
    return 0;
}

/* DirectML: a D3D11 device for libaji (aji_dml opens the shared frame
 * textures on a D3D12 device on the same adapter) and for libav's D3D11
 * frame pools, uploads and readbacks. Decoding stays on the CPU: hardware
 * decoder textures can't be shared to D3D12. */
static int setup_d3d11(enc_ctx *c)
{
#ifdef _WIN32
    if (av_hwdevice_ctx_create(&c->hw_device, AV_HWDEVICE_TYPE_D3D11VA,
                               NULL, NULL, 0) < 0) {
        loge("av_hwdevice_ctx_create(D3D11VA) failed");
        return -1;
    }
    AVHWDeviceContext      *dc = (AVHWDeviceContext *)c->hw_device->data;
    AVD3D11VADeviceContext *dx = (AVD3D11VADeviceContext *)dc->hwctx;
    c->d3d11_dev = dx->device;
    return 0;
#else
    (void)c;
    loge("the DirectML backend is Windows-only");
    return -1;
#endif
}

/* ---- input + decoder ----------------------------------------------------- */

static int open_input(enc_ctx *c)
{
    AV(avformat_open_input(&c->ifmt, c->o.input, NULL, NULL));
    AV(avformat_find_stream_info(c->ifmt, NULL));

    const AVCodec *vcodec = NULL;
    int vidx = av_find_best_stream(c->ifmt, AVMEDIA_TYPE_VIDEO, -1, -1,
                                   &vcodec, 0);
    if (vidx < 0) { loge("no video stream found"); goto fail; }
    c->vstream = vidx;

    AVStream *vst = c->ifmt->streams[vidx];
    AVCodecParameters *par = vst->codecpar;
    c->src_w = par->width;
    c->src_h = par->height;

    /* input aji_frame.format mirrors the source bit depth (NVDEC emits NV12
     * for 8-bit, P010 for 10-bit). The OUTPUT format is decoupled and set by
     * resolve_pixfmt() from --pix-fmt. */
    const AVPixFmtDescriptor *d = av_pix_fmt_desc_get(par->format);
    if (d && (d->log2_chroma_w != 1 || d->log2_chroma_h != 1))
        c->src_aji_fmt = AJI_FMT_YUV444P16;  /* 4:2:2 / 4:4:4: keep chroma */
    else
        c->src_aji_fmt = (d && d->comp[0].depth > 8) ? AJI_FMT_P010
                                                      : AJI_FMT_NV12;

    c->color_space = par->color_space;
    c->color_range = par->color_range;
    c->color_pri   = par->color_primaries;
    c->color_trc   = par->color_trc;
    c->chroma_loc  = par->chroma_location;

    c->out_fps = av_guess_frame_rate(c->ifmt, vst, NULL);
    if (c->out_fps.num == 0) c->out_fps = (AVRational){24000, 1001};

    /* frame-count estimate for progress */
    if (vst->nb_frames > 0)
        c->src_frames_est = vst->nb_frames;
    else if (c->ifmt->duration > 0)
        c->src_frames_est = (int64_t)((double)c->ifmt->duration / AV_TIME_BASE
                            * av_q2d(c->out_fps) + 0.5);
    return 0;
fail:
    return -1;
}

/* Resolve the output chroma + bit depth from --pix-fmt. Sets the final
 * encoder pix_fmt + flags, then picks the libaji output format and CUDA pool
 * sw_format.
 *
 * libaji constraint: aji_infer's output format must equal the input format or
 * be YUV444P16 (it won't, e.g., turn NV12 into P010). So we use a direct
 * NV12/P010 output only when it exactly matches the source 4:2:0 format
 * (enabling NVENC zero-copy); for every other request (bit-depth change,
 * chroma change, or 4:4:4) we route through libaji's 16-bit 4:4:4
 * intermediate and downconvert to the target with swscale on the host. */
static void resolve_pixfmt(enc_ctx *c)
{
    int nvenc = c->o.vcodec && strstr(c->o.vcodec, "nvenc");
    switch (c->o.pix_fmt) {
    case OUT_420P8:  c->ten_bit=0; c->is_444=0; break;
    case OUT_420P10: c->ten_bit=1; c->is_444=0; break;
    case OUT_444P8:  c->ten_bit=0; c->is_444=1; break;
    case OUT_444P10: c->ten_bit=1; c->is_444=1; break;
    }
    /* Encoder-input (and swscale-target) format. NVENC accepts only the
     * semi-planar 4:2:0 host formats (nv12/p010le), not planar yuv420p*;
     * software encoders take the planar forms. 4:4:4 is software-only. */
    if (c->is_444)
        c->out_pixfmt = c->ten_bit ? AV_PIX_FMT_YUV444P10LE : AV_PIX_FMT_YUV444P;
    else if (nvenc)
        c->out_pixfmt = c->ten_bit ? AV_PIX_FMT_P010LE : AV_PIX_FMT_NV12;
    else
        c->out_pixfmt = c->ten_bit ? AV_PIX_FMT_YUV420P10LE : AV_PIX_FMT_YUV420P;

    /* libaji output format + CUDA pool sw_format, chosen purely by --pix-fmt
     * (libaji converts any 4:2:0 input to NV12/P010 on-GPU, so 4:2:0 outputs
     * are zero-copy regardless of source). Only 4:4:4 uses the YUV444P16
     * intermediate (downloaded + downconverted by the software path). */
    switch (c->o.pix_fmt) {
    case OUT_420P8:
        c->out_aji_fmt = AJI_FMT_NV12;      c->sw_fmt = AV_PIX_FMT_NV12;        break;
    case OUT_420P10:
        c->out_aji_fmt = AJI_FMT_P010;      c->sw_fmt = AV_PIX_FMT_P010LE;      break;
    case OUT_444P8:
    case OUT_444P10:
        c->out_aji_fmt = AJI_FMT_YUV444P16; c->sw_fmt = AV_PIX_FMT_YUV444P16LE; break;
    }
    /* DirectML writes the output in the input's format (4:2:0 only), so the
     * host repack before upload produces the output format directly: an
     * 8-bit source for a 10-bit file is widened there, and 4:2:2 / 4:4:4
     * sources are subsampled. */
    if (c->dml)
        c->src_aji_fmt = c->out_aji_fmt;
}

static int open_decoder(enc_ctx *c)
{
    AVStream *vst = c->ifmt->streams[c->vstream];
    const AVCodec *codec = avcodec_find_decoder(vst->codecpar->codec_id);
    if (!codec) { loge("no decoder for video stream"); return -1; }

    c->dec = avcodec_alloc_context3(codec);
    if (!c->dec) { loge("avcodec_alloc_context3 failed"); return -1; }
    AV(avcodec_parameters_to_context(c->dec, vst->codecpar));
    c->dec->pkt_timebase = vst->time_base;

    int want_hw = (c->o.decoder != DEC_CPU) && !c->o.no_zerocopy && !c->dml;
    if (want_hw) {
        c->dec->hw_device_ctx = av_buffer_ref(c->hw_device);
        c->dec->get_format = get_hw_format;
        c->dec->opaque = c;
    }
    int r = avcodec_open2(c->dec, codec, NULL);
    if (r < 0 && want_hw && c->o.decoder == DEC_AUTO) {
        loge("NVDEC open failed; falling back to CPU decode");
        av_buffer_unref(&c->dec->hw_device_ctx);
        c->dec->get_format = NULL;
        avcodec_free_context(&c->dec);
        c->dec = avcodec_alloc_context3(codec);
        AV(avcodec_parameters_to_context(c->dec, vst->codecpar));
        c->dec->pkt_timebase = vst->time_base;
        want_hw = 0;
        r = avcodec_open2(c->dec, codec, NULL);
    }
    if (r < 0) { AV(r); }
    c->nvdec = want_hw;
    return 0;
fail:
    return -1;
}

/* ---- libaji init --------------------------------------------------------- */

static int init_aji(enc_ctx *c)
{
    aji_create_params p = {
        .api_version    = AJI_API_VERSION,
        .cuda_context   = (void *)c->cuctx,
        .d3d11_device   = c->d3d11_dev,
        .conf_path      = c->o.conf,
        .model_dir      = c->o.model_dir,
        .trtexec        = c->o.trtexec,
        .trtexec_env    = c->o.trtexec_env,
        .slot           = c->o.slot,
        .rife_model_dir = c->o.rife_model_dir,
        .async_build    = 0,     /* CLI: block on engine builds */
        .engine_path    = c->o.engine,
        .max_width      = c->src_w,
        .max_height     = c->src_h,
        .log            = aji_log_cb,
    };
    c->aji = aji_create(&p);
    if (!c->aji) { loge("aji_create failed"); return -1; }

    progress("build_engine", 0, 0, 0, 0);
    int ow = 0, oh = 0;
    int act = aji_configure(c->aji, c->src_w, c->src_h, av_q2d(c->out_fps),
                            &ow, &oh);
    if (act < 0) {
        loge("aji_configure: %d: %s", act, aji_last_error(c->aji));
        return -1;
    }
    progress("build_engine", 0, 0, 0, 100);
    loge("%s", aji_current_log(c->aji));

    if (act == 0) {
        c->passthrough = 1;
        c->up_w = c->src_w;
        c->up_h = c->src_h;
        loge("no chain active for %dx%d @ %.3f fps; transcoding (passthrough)",
             c->src_w, c->src_h, av_q2d(c->out_fps));
    } else {
        c->up_w = ow;
        c->up_h = oh;
        c->temporal_r = aji_temporal_radius(c->aji);
        if (c->temporal_r < 0) c->temporal_r = 0;
        if (c->temporal_r > 0)
            loge("temporal chain: %d-frame window", 2 * c->temporal_r + 1);
    }
    /* final encode dims default to the upscale dims; --final-resize adjusts */
    c->out_w = c->up_w;
    c->out_h = c->up_h;

    if (aji_rife_factor(c->aji, &c->rnum, &c->rden) && c->rnum > 0) {
        if (c->rden != 1 || c->rnum % c->rden != 0) {
            loge("rational RIFE factor %d/%d not yet supported in offline "
                 "encode (v1 handles integer factors only)", c->rnum, c->rden);
            return -1;
        }
        c->rife = 1;
        c->out_fps = av_mul_q(c->out_fps, (AVRational){c->rnum, c->rden});
        if (c->passthrough) {
            /* RIFE-only chain (no scaling): the engine interpolates the
             * passthrough frames, so they go to the device like any input -
             * already in the output format, which RIFE keeps - and no upscale
             * runs, whatever the order setting says */
            c->rife_only = 1;
            c->src_aji_fmt = c->out_aji_fmt;
            loge("RIFE-only chain: interpolating at %dx%d", c->src_w, c->src_h);
            return 0;
        }
        /* the order must agree with what the engine configured RIFE for */
        c->rife_first = aji_rife_before_upscale(c->aji) == 1;
        c->work_w = c->src_w;
        c->work_h = c->src_h;
        int ww = 0, wh = 0;
        if (c->rife_first && aji_pre_resize(c->aji, &ww, &wh) &&
            ww > 0 && wh > 0) {
            c->pre_resize = 1;
            c->work_w = ww;
            c->work_h = wh;
        }
        if (c->rife_first)
            loge("RIFE-first: interpolating at %dx%d, then upscaling",
                 c->work_w, c->work_h);
    }
    return 0;
}

/* A frame pool on the active backend's device: CUDA frames, or (DirectML)
 * D3D11 textures created SHARED | SHARED_NTHANDLE so aji_dml can open them
 * on its D3D12 device. libav's D3D11 frame data[0]/data[1] (texture,
 * subresource index) is exactly aji_frame.plane[0]/plane[1] for DirectML,
 * so the rest of the pipeline handles both alike. CUDA pools are fixed-size
 * (n); D3D11 ones grow on demand, one texture per frame like mpv's. */
static AVBufferRef *make_pool(enc_ctx *c, enum AVPixelFormat sw, int w, int h,
                              int n)
{
    AVBufferRef *ref = av_hwframe_ctx_alloc(c->hw_device);
    if (!ref) { loge("av_hwframe_ctx_alloc failed"); return NULL; }
    AVHWFramesContext *fc = (AVHWFramesContext *)ref->data;
    fc->sw_format         = sw;
    fc->width             = w;
    fc->height            = h;
    if (c->dml) {
#ifdef _WIN32
        fc->format = AV_PIX_FMT_D3D11;
        AVD3D11VAFramesContext *dx = (AVD3D11VAFramesContext *)fc->hwctx;
        dx->BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        dx->MiscFlags = D3D11_RESOURCE_MISC_SHARED |
                        D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
#endif
    } else {
        fc->format            = AV_PIX_FMT_CUDA;
        fc->initial_pool_size = n;
    }
    if (av_hwframe_ctx_init(ref) < 0) {
        loge("%s frame pool init failed for %dx%d %s", c->dml ? "D3D11" : "CUDA",
             w, h, av_get_pix_fmt_name(sw));
        av_buffer_unref(&ref);
    }
    return ref;
}

/* Allocate the frame pool that libaji writes its upscaled output into.
 * CUDA frames come from a libav hw_frames_ctx so the device pointers are
 * valid for NVENC (zero-copy) and carry a hw_frames_ctx for scale_cuda. */
static int init_aji_pool(enc_ctx *c)
{
    /* cover in-flight ring/RIFE frames plus those the encoder holds
     * (NVENC async depth + B-frames) — the CUDA pool is fixed-size */
    c->aji_pool = make_pool(c, c->sw_fmt, c->up_w, c->up_h, c->ring_depth + 48);
    return c->aji_pool ? 0 : -1;
}

/* ---- map a decoded AVFrame into an aji_frame ----------------------------- */

static void fill_aji_frame(enc_ctx *c, aji_frame *f, AVFrame *av, int output)
{
    f->width  = av->width;
    f->height = av->height;
    /* input mirrors the source format; output uses the --pix-fmt selection */
    f->format = output ? c->out_aji_fmt : c->src_aji_fmt;
    f->matrix = aji_matrix_from_av(c->color_space);
    f->range  = aji_range_from_av(c->color_range);
    f->siting = aji_siting_from_av(c->chroma_loc);
    f->plane[0] = av->data[0];
    f->plane[1] = av->data[1];
    f->plane[2] = av->data[2];
    f->stride[0] = av->linesize[0];
    f->stride[1] = av->linesize[1];
    f->stride[2] = av->linesize[2];
}

/* device pool sw_format for frames in the libaji input format */
static enum AVPixelFormat src_sw_fmt(const enc_ctx *c)
{
    return c->src_aji_fmt == AJI_FMT_YUV444P16 ? AV_PIX_FMT_YUV444P16LE
         : c->src_aji_fmt == AJI_FMT_P010      ? AV_PIX_FMT_P010LE
                                                : AV_PIX_FMT_NV12;
}

/* Hand libaji a device frame in its input format. NVDEC frames already are
 * one (zero-copy); CPU-decoded frames (always, on DirectML), and NVDEC
 * frames in a layout libaji does not read, are repacked on the host and
 * uploaded. Takes ownership. */
static AVFrame *to_device(enc_ctx *c, AVFrame *fr)
{
    const enum AVPixelFormat sw = src_sw_fmt(c);
    if (fr->format == AV_PIX_FMT_CUDA && fr->hw_frames_ctx &&
        ((AVHWFramesContext *)fr->hw_frames_ctx->data)->sw_format == sw)
        return fr;

    AVFrame *host = NULL, *packed = NULL, *dev = NULL;
    if (!c->src_pool &&
        !(c->src_pool = make_pool(c, sw, c->src_w, c->src_h,
                                  c->ring_depth + 8)))
        goto fail;
    if (fr->format == AV_PIX_FMT_CUDA) {
        host = av_frame_alloc();
        if (!host || av_hwframe_transfer_data(host, fr, 0) < 0) {
            loge("download of a %s NVDEC frame failed", av_get_pix_fmt_name(
                 ((AVHWFramesContext *)fr->hw_frames_ctx->data)->sw_format));
            goto fail;
        }
    }
    const AVFrame *src = host ? host : fr;
    if (src->format != sw) {
        c->in_sws = sws_getCachedContext(c->in_sws, src->width, src->height,
                                         src->format, src->width, src->height,
                                         sw, SWS_BICUBIC, NULL, NULL, NULL);
        packed = av_frame_alloc();
        if (!c->in_sws || !packed) goto fail;
        packed->format = sw;
        packed->width  = src->width;
        packed->height = src->height;
        if (av_frame_get_buffer(packed, 0) < 0 ||
            sws_scale_frame(c->in_sws, packed, src) < 0) {
            loge("source repack %s -> %s failed",
                 av_get_pix_fmt_name(src->format), av_get_pix_fmt_name(sw));
            goto fail;
        }
        src = packed;
    }
    dev = av_frame_alloc();
    if (!dev || av_hwframe_get_buffer(c->src_pool, dev, 0) < 0 ||
        av_hwframe_transfer_data(dev, src, 0) < 0) {
        loge("source upload failed");
        goto fail;
    }
    av_frame_copy_props(dev, fr);
    av_frame_free(&packed);
    av_frame_free(&host);
    av_frame_free(&fr);
    return dev;
fail:
    av_frame_free(&dev);
    av_frame_free(&packed);
    av_frame_free(&host);
    av_frame_free(&fr);
    return NULL;
}

/* ---- output: encoder, muxer, streams ------------------------------------- */

static int is_nvenc(const char *name)
{
    return name && strstr(name, "nvenc") != NULL;
}

/* Parse the raw --vquality arg string ("-preset p7 -b:v 50M") into a dict.
 * Keys map to AVOptions on the encoder context (incl. private opts), which
 * avcodec_open2 parses for us (size suffixes, profile names, ...). */
/* portable whitespace tokenizer (no strtok_r/strtok_s): returns the next
 * token start, NUL-terminates it, and advances *p past it. */
static char *next_token(char **p)
{
    char *s = *p;
    while (*s == ' ' || *s == '\t') s++;
    if (!*s) { *p = s; return NULL; }
    char *start = s;
    while (*s && *s != ' ' && *s != '\t') s++;
    if (*s) { *s = 0; s++; }
    *p = s;
    return start;
}

static void parse_vquality(const char *s, AVDictionary **d)
{
    if (!s) return;
    char *buf = av_strdup(s);
    if (!buf) return;
    char *cur = buf, *tok;
    while ((tok = next_token(&cur))) {
        if (tok[0] != '-')
            continue;
        char key[64];
        snprintf(key, sizeof key, "%s", tok + 1);
        char *colon = strchr(key, ':');   /* "b:v" -> "b" */
        if (colon) *colon = 0;
        char *val = next_token(&cur);
        av_dict_set(d, key, val ? val : "1", 0);
    }
    av_free(buf);
}

/* Pick the encoder profile required for the chosen pix_fmt, unless the caller
 * already set one in --vquality. ffv1 needs none. */
static void inject_profile(const char *vcodec, enum out_pixfmt pf,
                           AVDictionary **opts)
{
    if (av_dict_get(*opts, "profile", 0, 0))
        return;
    const char *p = NULL;
    if (!strcmp(vcodec, "libx265")) {
        p = pf == OUT_420P10 ? "main10" :
            pf == OUT_444P8  ? "main444-8" :
            pf == OUT_444P10 ? "main444-10" : NULL;
    } else if (!strcmp(vcodec, "libx264")) {
        p = pf == OUT_420P10 ? "high10" :
            (pf == OUT_444P8 || pf == OUT_444P10) ? "high444" : NULL;
    } else if (strstr(vcodec, "nvenc")) {
        p = pf == OUT_420P10 ? "main10" : NULL;   /* nvenc is 4:2:0 only here */
    }
    if (p)
        av_dict_set(opts, "profile", p, 0);
}

static int open_encoder(enc_ctx *c, AVFrame *first)
{
    const AVCodec *codec = avcodec_find_encoder_by_name(c->o.vcodec);
    if (!codec) { loge("unknown --vcodec %s", c->o.vcodec); return -1; }

    c->enc = avcodec_alloc_context3(codec);
    if (!c->enc) { loge("encoder alloc failed"); return -1; }

    int w = first->width, h = first->height;
    c->enc->width  = w;
    c->enc->height = h;
    c->out_tb = av_inv_q(c->out_fps);
    c->enc->time_base = c->out_tb;
    c->enc->framerate = c->out_fps;
    c->enc->colorspace            = c->color_space;
    c->enc->color_range           = c->color_range;
    c->enc->color_primaries       = c->color_pri;
    c->enc->color_trc             = c->color_trc;
    c->enc->chroma_sample_location = c->chroma_loc;

    int hw = is_nvenc(c->o.vcodec) && first->hw_frames_ctx &&
             (first->format == AV_PIX_FMT_CUDA ||
              first->format == AV_PIX_FMT_D3D11);
    if (hw) {
        c->enc->pix_fmt = first->format;
        c->enc->hw_frames_ctx = av_buffer_ref(first->hw_frames_ctx);
        if (!c->enc->hw_frames_ctx) { loge("no hw_frames_ctx"); return -1; }
    } else {
        c->enc->pix_fmt = c->out_pixfmt;
    }
    if (c->ofmt && (c->ofmt->oformat->flags & AVFMT_GLOBALHEADER))
        c->enc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    AVDictionary *opts = NULL;
    parse_vquality(c->o.vquality, &opts);
    inject_profile(c->o.vcodec, c->o.pix_fmt, &opts);

    int r = avcodec_open2(c->enc, codec, &opts);
    AVDictionaryEntry *e = NULL;
    while ((e = av_dict_get(opts, "", e, AV_DICT_IGNORE_SUFFIX)))
        loge("warning: --vquality option '%s=%s' not recognized by %s",
             e->key, e->value, c->o.vcodec);
    av_dict_free(&opts);
    if (r < 0) { AV(r); }
    return 0;
fail:
    return -1;
}

/* Add the output muxer, the video stream, and stream-copy aux streams. */
static int open_output(enc_ctx *c)
{
    AV(avformat_alloc_output_context2(&c->ofmt, NULL, NULL, c->o.output));
    if (!c->ofmt->oformat) {
        AV(avformat_alloc_output_context2(&c->ofmt, NULL, "matroska",
                                          c->o.output));
    }

    /* video stream */
    AVStream *vs = avformat_new_stream(c->ofmt, NULL);
    if (!vs) { loge("new video stream failed"); return -1; }
    c->vout = vs->index;

    /* aux streams (stream-copy) */
    c->smap = av_malloc_array(c->ifmt->nb_streams, sizeof(int));
    for (unsigned i = 0; i < c->ifmt->nb_streams; i++) c->smap[i] = -1;
    for (unsigned i = 0; i < c->ifmt->nb_streams; i++) {
        if ((int)i == c->vstream) continue;
        AVStream *in = c->ifmt->streams[i];
        enum AVMediaType t = in->codecpar->codec_type;
        if (t == AVMEDIA_TYPE_AUDIO && c->o.no_audio) continue;
        if (t == AVMEDIA_TYPE_SUBTITLE && c->o.no_subs) continue;
        if (t != AVMEDIA_TYPE_AUDIO && t != AVMEDIA_TYPE_SUBTITLE &&
            t != AVMEDIA_TYPE_ATTACHMENT && t != AVMEDIA_TYPE_DATA)
            continue;
        AVStream *out = avformat_new_stream(c->ofmt, NULL);
        if (!out) { loge("new aux stream failed"); return -1; }
        AV(avcodec_parameters_copy(out->codecpar, in->codecpar));
        out->codecpar->codec_tag = 0;
        out->time_base = in->time_base;
        av_dict_copy(&out->metadata, in->metadata, 0);
        c->smap[i] = out->index;
    }

    /* chapters + global metadata */
    if (!c->o.no_chapters && c->ifmt->nb_chapters) {
        for (unsigned i = 0; i < c->ifmt->nb_chapters; i++) {
            AVChapter *ic = c->ifmt->chapters[i];
            AVChapter *oc = av_mallocz(sizeof(*oc));
            if (!oc) break;
            *oc = *ic;
            oc->metadata = NULL;
            av_dict_copy(&oc->metadata, ic->metadata, 0);
            AVChapter **tmp = av_realloc_f(c->ofmt->chapters,
                                           c->ofmt->nb_chapters + 1,
                                           sizeof(*tmp));
            if (!tmp) { av_free(oc); break; }
            c->ofmt->chapters = tmp;
            c->ofmt->chapters[c->ofmt->nb_chapters++] = oc;
        }
    }
    av_dict_copy(&c->ofmt->metadata, c->ifmt->metadata, 0);
    return 0;
fail:
    return -1;
}

/* Finish output setup once the first encoded video frame is known: copy the
 * encoder params onto the video stream, open the file, write the header, and
 * flush any aux packets buffered before this point. */
static int finalize_output(enc_ctx *c)
{
    AVStream *vs = c->ofmt->streams[c->vout];
    AV(avcodec_parameters_from_context(vs->codecpar, c->enc));
    vs->time_base = c->enc->time_base;
    vs->avg_frame_rate = c->out_fps;

    if (!(c->ofmt->oformat->flags & AVFMT_NOFILE)) {
        if (!c->o.overwrite) {
            FILE *t = fopen(c->o.output, "rb");
            if (t) { fclose(t); loge("output exists (use --overwrite)"); return -1; }
        }
        AV(avio_open(&c->ofmt->pb, c->o.output, AVIO_FLAG_WRITE));
    }

    AVDictionary *mopt = NULL;
    av_dict_set_int(&mopt, "max_interleave_delta", 0, 0);
    int r = avformat_write_header(c->ofmt, &mopt);
    av_dict_free(&mopt);
    if (r < 0) { AV(r); }
    c->opened = 1;

    for (int i = 0; i < c->aux_n; i++) {
        av_interleaved_write_frame(c->ofmt, c->aux_buf[i]);
        av_packet_free(&c->aux_buf[i]);
    }
    c->aux_n = 0;
    return 0;
fail:
    return -1;
}

/* ---- optional final resize (nvenc path): scale_cuda filtergraph ---------- */

static int init_resize(enc_ctx *c, AVFrame *first)
{
    if (!c->graph) {
        if (!avfilter_get_by_name("scale_cuda")) {
            loge("--final-resize requested but scale_cuda filter unavailable");
            return -1;
        }
        c->graph = avfilter_graph_alloc();
        if (!c->graph) return -1;

        char args[512];
        AVStream *vst = c->ifmt->streams[c->vstream];
        snprintf(args, sizeof args,
                 "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=1/1",
                 first->width, first->height, AV_PIX_FMT_CUDA,
                 vst->time_base.num, vst->time_base.den);
        AV(avfilter_graph_create_filter(&c->buf_src,
                avfilter_get_by_name("buffer"), "in", args, NULL, c->graph));

        AVBufferSrcParameters *bp = av_buffersrc_parameters_alloc();
        bp->hw_frames_ctx = first->hw_frames_ctx;
        AV(av_buffersrc_parameters_set(c->buf_src, bp));
        av_free(bp);

        AV(avfilter_graph_create_filter(&c->buf_sink,
                avfilter_get_by_name("buffersink"), "out", NULL, NULL,
                c->graph));

        char sargs[128];
        snprintf(sargs, sizeof sargs, "w=%d:h=%d", c->out_w, c->out_h);
        AVFilterContext *scale = NULL;
        AV(avfilter_graph_create_filter(&scale,
                avfilter_get_by_name("scale_cuda"), "scale", sargs, NULL,
                c->graph));
        AV(avfilter_link(c->buf_src, 0, scale, 0));
        AV(avfilter_link(scale, 0, c->buf_sink, 0));
        AV(avfilter_graph_config(c->graph, NULL));
    }
    return 0;
fail:
    return -1;
}

/* Compute final (post-resize) dims from the upscale output dims + opts. */
static void compute_final_dims(enc_ctx *c)
{
    if (c->o.final_h > 0) {
        int h = c->o.final_h & ~1;
        int w = (int)((int64_t)c->out_w * h / c->out_h) & ~1;
        c->out_w = w; c->out_h = h;
    } else if (c->o.final_pct > 0 && c->o.final_pct != 100) {
        c->out_w = ((c->out_w * c->o.final_pct / 100) + 1) & ~1;
        c->out_h = ((c->out_h * c->o.final_pct / 100) + 1) & ~1;
    }
}

static int resize_needed(enc_ctx *c)
{
    return c->o.final_h > 0 || (c->o.final_pct > 0 && c->o.final_pct != 100);
}

/* ---- emit one finished output frame: resize -> encode -> mux ------------- */

/* Emit one finished output frame. Frames arrive in display order (the
 * non-RIFE ring drains FIFO; RIFE emits base then phases in order), so we
 * assign the monotonic CFR PTS here and own the emitted-frame count. */
static int emit_output(enc_ctx *c, AVFrame *cuda_out)
{
    int64_t pts = c->out_frames;
    AVFrame *to_encode = NULL;
    AVFrame *resized = NULL;
    int rc = -1;

    /* Zero-copy only when the libaji output is a CUDA NV12/P010 frame NVENC
     * can read directly. Anything routed through the YUV444P16 intermediate
     * (bit-depth/chroma change or 4:4:4) is downloaded and converted with
     * swscale, then handed to the encoder (NVENC uploads host frames itself;
     * software encoders consume them directly). DirectML's D3D11 output goes
     * to NVENC zero-copy too (NVENC takes D3D11 textures), except with a
     * final resize (scale_cuda is CUDA-only); every other encoder reads it
     * back. */
    int zerocopy = is_nvenc(c->o.vcodec) &&
                   c->out_aji_fmt != AJI_FMT_YUV444P16 &&
                   (!c->dml || !resize_needed(c));

    if (zerocopy && resize_needed(c)) {
        if (init_resize(c, cuda_out) < 0) goto done;
        AV(av_buffersrc_add_frame_flags(c->buf_src, cuda_out,
                                        AV_BUFFERSRC_FLAG_KEEP_REF));
        resized = av_frame_alloc();
        AV(av_buffersink_get_frame(c->buf_sink, resized));
        to_encode = resized;
    } else if (zerocopy) {
        to_encode = cuda_out;          /* zero-copy CUDA frame straight in */
    } else {
        /* software encode: download to host then convert/resize via swscale.
         * The encoder isn't open yet on the first frame, so derive the target
         * pixfmt/dims from the context, not from c->enc. */
        AVFrame *host = cuda_out;
        AVFrame *dl = NULL;
        if (cuda_out->hw_frames_ctx) {     /* CUDA, or DirectML's D3D11 */
            dl = av_frame_alloc();
            AV(av_hwframe_transfer_data(dl, cuda_out, 0));
            host = dl;
        }
        if (host->format == c->out_pixfmt && host->width == c->out_w &&
            host->height == c->out_h) {
            /* already what the encoder takes (e.g. DirectML NV12/P010 into
             * NVENC): skip a full-frame swscale copy */
            to_encode = host;
            resized = dl;              /* freed below; NULL = caller's frame */
            goto encode;
        }
        AVFrame *sw = av_frame_alloc();
        sw->format = c->out_pixfmt;
        sw->width  = c->out_w;
        sw->height = c->out_h;
        AV(av_frame_get_buffer(sw, 0));
        c->sws = sws_getCachedContext(c->sws, host->width, host->height,
                    host->format, sw->width, sw->height, sw->format,
                    SWS_BICUBIC, NULL, NULL, NULL);
        if (!c->sws) { loge("sws_getContext failed"); av_frame_free(&sw);
                       av_frame_free(&dl); goto done; }
        sws_scale(c->sws, (const uint8_t *const *)host->data, host->linesize,
                  0, host->height, sw->data, sw->linesize);
        av_frame_free(&dl);
        to_encode = sw;
        resized = sw;                  /* freed below */
    }

encode:
    if (!c->opened) {
        if (open_encoder(c, to_encode) < 0) goto done;
        if (finalize_output(c) < 0) goto done;
    }

    /* Clone before sending so the queued frame owns its own pts (callers
     * reuse frame objects across emits — e.g. RIFE scene-dup) and the
     * underlying buffer stays ref-counted for the encoder's lifetime. */
    {
        AVFrame *sendf = av_frame_clone(to_encode);
        if (!sendf) { loge("av_frame_clone failed"); goto done; }
        sendf->pts = pts;
        int sret = avcodec_send_frame(c->enc, sendf);
        av_frame_free(&sendf);
        if (sret < 0) { AV(sret); }
    }
    if (drain_encoder(c, 0) < 0) goto done;

    c->out_frames++;
    if ((c->out_frames & 31) == 0) {
        int pct = c->src_frames_est ? (int)(c->out_frames * 100 /
                  (c->src_frames_est * (c->rife ? c->rnum : 1) /
                   (c->rife ? c->rden : 1))) : 0;
        progress("encode", c->out_frames,
                 c->src_frames_est * (c->rife ? c->rnum : 1) /
                 (c->rife ? c->rden : 1), 0, pct > 100 ? 100 : pct);
    }
    rc = 0;
done:
    av_frame_free(&resized);
    return rc;
fail:
    av_frame_free(&resized);
    return -1;
}

static int drain_encoder(enc_ctx *c, int flush)
{
    if (flush)
        avcodec_send_frame(c->enc, NULL);
    for (;;) {
        AVPacket *pkt = av_packet_alloc();
        int r = avcodec_receive_packet(c->enc, pkt);
        if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) {
            av_packet_free(&pkt);
            return 0;
        }
        if (r < 0) { av_packet_free(&pkt); AV(r); }
        pkt->stream_index = c->vout;
        av_packet_rescale_ts(pkt, c->enc->time_base,
                             c->ofmt->streams[c->vout]->time_base);
        r = av_interleaved_write_frame(c->ofmt, pkt);
        av_packet_free(&pkt);
        if (r < 0) { AV(r); }
    }
fail:
    return -1;
}

/* ---- non-RIFE pipelined ring --------------------------------------------- */

static int ring_reclaim(enc_ctx *c, slot *s)
{
    if (s->ticket) {
        if (aji_wait(c->aji, s->ticket) != AJI_OK) {
            loge("aji_wait failed: %s", aji_last_error(c->aji));
            return -1;
        }
    }
    int r = emit_output(c, s->out);
    av_frame_free(&s->out);
    av_frame_free(&s->in_ref);
    s->ticket = 0;
    return r;
}

/* reclaim the oldest in-flight slot once the ring is full */
static int ring_make_room(enc_ctx *c)
{
    if (c->ring_count == c->ring_depth) {
        slot *s = &c->ring[(c->ring_head - c->ring_count + MAX_RING)
                           % MAX_RING];
        if (ring_reclaim(c, s) < 0) return -1;
        c->ring_count--;
    }
    return 0;
}

static int ring_drain(enc_ctx *c)
{
    while (c->ring_count > 0) {
        slot *s = &c->ring[(c->ring_head - c->ring_count + MAX_RING)
                           % MAX_RING];
        if (ring_reclaim(c, s) < 0) return -1;
        c->ring_count--;
    }
    return 0;
}

static void ring_commit(enc_ctx *c, AVFrame *in_ref, AVFrame *out,
                        uint64_t ticket)
{
    slot *s = &c->ring[c->ring_head];
    s->in_ref = in_ref;
    s->out = out;
    s->ticket = ticket;
    c->ring_head = (c->ring_head + 1) % MAX_RING;
    c->ring_count++;
}

/* a pool frame for libaji to write an upscaled frame into */
static AVFrame *alloc_up(enc_ctx *c)
{
    AVFrame *out = av_frame_alloc();
    if (av_hwframe_get_buffer(c->aji_pool, out, 0) < 0) {
        loge("hwframe_get_buffer failed"); av_frame_free(&out);
        return NULL;
    }
    out->color_range = c->color_range;
    out->colorspace  = c->color_space;
    out->chroma_location = c->chroma_loc;
    return out;
}

/* block until everything enqueued on the stream so far has finished */
static int sync_aji(enc_ctx *c)
{
    uint64_t t = aji_flush(c->aji, c->stream);
    if (t && aji_wait(c->aji, t) != AJI_OK) {
        loge("aji_wait failed: %s", aji_last_error(c->aji));
        return -1;
    }
    return 0;
}

/* temporal chains: add one decoded frame to libaji's frame history */
static int ingest(enc_ctx *c, AVFrame *dec)
{
    aji_frame in_f;
    fill_aji_frame(c, &in_f, dec, 0);
    int ret = aji_ingest(c->aji, &in_f, c->in_seq, c->stream);
    if (ret != AJI_OK) {
        loge("aji_ingest: %d: %s", ret, aji_last_error(c->aji));
        return -1;
    }
    c->in_seq++;
    return 0;
}

/* temporal chains: enqueue the upscale of the next frame due (out_seq) */
static AVFrame *infer_next_seq(enc_ctx *c)
{
    AVFrame *out = alloc_up(c);
    if (!out) return NULL;
    aji_frame out_f;
    fill_aji_frame(c, &out_f, out, 1);
    int ret = aji_infer_seq(c->aji, c->out_seq, &out_f, c->stream);
    if (ret != AJI_OK) {
        loge("aji_infer_seq: %d: %s", ret, aji_last_error(c->aji));
        av_frame_free(&out);
        return NULL;
    }
    c->out_seq++;
    return out;
}

/* temporal chain, non-RIFE (dec==NULL flushes). Frame k is inferred once
 * k+r is ingested, so past the first r frames each push pairs one ingest
 * with one inference; the slot keeps the frame ingested with it alive until
 * its ticket completes. The last r frames drain at EOF with a clamped
 * window. */
static int push_upscale_temporal(enc_ctx *c, AVFrame *dec)
{
    if (dec) {
        int r = ingest(c, dec);
        if (r == 0 && c->in_seq <= (uint64_t)c->temporal_r)
            r = sync_aji(c);           /* warm-up: nothing is due yet */
        if (r < 0 || c->in_seq <= (uint64_t)c->temporal_r) {
            av_frame_free(&dec);
            return r;
        }
        if (ring_make_room(c) < 0) { av_frame_free(&dec); return -1; }
        AVFrame *out = infer_next_seq(c);
        if (!out) { av_frame_free(&dec); return -1; }
        ring_commit(c, dec, out, aji_flush(c->aji, c->stream));
        return 0;
    }
    while (c->out_seq < c->in_seq) {
        if (ring_make_room(c) < 0) return -1;
        AVFrame *out = infer_next_seq(c);
        if (!out) return -1;
        ring_commit(c, NULL, out, aji_flush(c->aji, c->stream));
    }
    return ring_drain(c);
}

/* push one decoded frame (dec==NULL flushes the ring) */
static int push_upscale(enc_ctx *c, AVFrame *dec)
{
    if (c->temporal_r) return push_upscale_temporal(c, dec);
    if (!dec) return ring_drain(c);
    if (ring_make_room(c) < 0) return -1;

    if (c->passthrough) {
        /* same frame; freed via s->out */
        ring_commit(c, NULL, dec, 0);
        return 0;
    }

    AVFrame *out = alloc_up(c);
    if (!out) return -1;
    aji_frame in_f, out_f;
    fill_aji_frame(c, &in_f, dec, 0);
    fill_aji_frame(c, &out_f, out, 1);
    int ret = aji_infer(c->aji, &in_f, &out_f, c->stream);
    if (ret != AJI_OK) {
        loge("aji_infer: %d: %s", ret, aji_last_error(c->aji));
        av_frame_free(&out);
        return -1;
    }
    ring_commit(c, dec, out, aji_flush(c->aji, c->stream));
    return 0;
}

/* ---- RIFE: synchronous upscale of neighbors + interpolation -------------- */

static AVFrame *upscale_one(enc_ctx *c, AVFrame *dec)
{
    AVFrame *out = alloc_up(c);
    if (!out) return NULL;
    aji_frame in_f, out_f;
    fill_aji_frame(c, &in_f, dec, 0);
    fill_aji_frame(c, &out_f, out, 1);
    int ret = aji_infer(c->aji, &in_f, &out_f, c->stream);
    if (ret != AJI_OK) {
        loge("aji_infer: %d: %s", ret, aji_last_error(c->aji));
        av_frame_free(&out);
        return NULL;
    }
    if (sync_aji(c) < 0) { av_frame_free(&out); return NULL; }
    return out;
}

/* take one upscaled frame (up==NULL flushes): emit the previous one and the
 * interpolation phases between the two */
static int rife_step(enc_ctx *c, AVFrame *up)
{
    if (!up) {
        /* emit the final upscaled frame, no successor to interpolate with */
        if (c->have_prev) {
            if (emit_output(c, c->up_prev) < 0) return -1;
        }
        return 0;
    }

    if (!c->have_prev) {
        c->up_prev = up;
        c->have_prev = 1;
        return 0;
    }
    c->up_cur = up;

    /* emit base frame N (= up_prev) */
    if (emit_output(c, c->up_prev) < 0) return -1;

    /* interpolation phases between N and N+1 */
    double factor = (double)c->rnum / c->rden;
    int phases = c->rnum / c->rden;        /* integer factor: F-1 phases */
    if (c->rnum % c->rden == 0) {
        for (int k = 1; k < phases; k++) {
            double t = (double)k / phases;
            /* fresh pool buffer per phase: the encoder holds a ref to the
             * previous one, so we must not overwrite it */
            AVFrame *interp = av_frame_alloc();
            if (av_hwframe_get_buffer(c->aji_pool, interp, 0) < 0) {
                loge("hwframe_get_buffer (interp) failed");
                av_frame_free(&interp); return -1;
            }
            aji_frame a, b, o;
            fill_aji_frame(c, &a, c->up_prev, 1);
            fill_aji_frame(c, &b, c->up_cur, 1);
            fill_aji_frame(c, &o, interp, 1);
            int ret = aji_infer_rife(c->aji, &a, &b, t, &o, c->stream);
            AVFrame *emit = interp;
            if (ret == AJI_SCENE) { emit = c->up_prev; c->scene_dupes++; }
            else if (ret != AJI_OK) {
                loge("aji_infer_rife: %d: %s", ret, aji_last_error(c->aji));
                av_frame_free(&interp); return -1;
            }
            int er = emit_output(c, emit);
            av_frame_free(&interp);
            if (er < 0) return -1;
        }
    } else {
        (void)factor;   /* rational factors: handled in a later revision */
    }

    av_frame_free(&c->up_prev);
    c->up_prev = c->up_cur;
    c->up_cur = NULL;
    return 0;
}

/* temporal chain + RIFE: the same ingest/infer schedule as
 * push_upscale_temporal, but each upscale is waited on before RIFE
 * interpolates against it */
static int rife_temporal_one(enc_ctx *c)
{
    AVFrame *up = infer_next_seq(c);
    if (!up) return -1;
    if (sync_aji(c) < 0) { av_frame_free(&up); return -1; }
    return rife_step(c, up);
}

static int push_rife(enc_ctx *c, AVFrame *dec)
{
    if (c->temporal_r) {
        if (!dec) {
            while (c->out_seq < c->in_seq)
                if (rife_temporal_one(c) < 0) return -1;
            return rife_step(c, NULL);
        }
        int r = ingest(c, dec);
        if (r == 0 && c->in_seq <= (uint64_t)c->temporal_r)
            r = sync_aji(c);           /* warm-up: nothing is due yet */
        if (r < 0 || c->in_seq <= (uint64_t)c->temporal_r) {
            av_frame_free(&dec);
            return r;
        }
        r = rife_temporal_one(c);      /* waits on the ingest too */
        av_frame_free(&dec);
        return r;
    }
    if (!dec) return rife_step(c, NULL);
    AVFrame *up = upscale_one(c, dec);
    av_frame_free(&dec);
    if (!up) return -1;
    return rife_step(c, up);
}

/* ---- RIFE-first: interpolate source frames, then upscale each ----------- */

/* a source-format frame at the work dims (downscales and interpolations) */
static AVFrame *alloc_work(enc_ctx *c)
{
    if (!c->work_pool &&
        !(c->work_pool = make_pool(c, src_sw_fmt(c), c->work_w, c->work_h,
                                   8)))
        return NULL;
    AVFrame *f = av_frame_alloc();
    if (av_hwframe_get_buffer(c->work_pool, f, 0) < 0) {
        loge("hwframe_get_buffer (work) failed"); av_frame_free(&f);
        return NULL;
    }
    return f;
}

/* RIFE-first (dec==NULL flushes): when source frame N+1 arrives, emit
 * upscaled N, then upscale each interpolation between the two. A scene
 * change repeats upscaled N, like the default order. */
static int push_rife_first(enc_ctx *c, AVFrame *dec)
{
    if (!dec) {
        if (!c->src_prev) return 0;
        AVFrame *up = upscale_one(c, c->src_prev);
        int r = up ? emit_output(c, up) : -1;
        av_frame_free(&up);
        return r;
    }

    AVFrame *cur = dec;
    if (c->pre_resize) {
        cur = alloc_work(c);
        if (!cur) { av_frame_free(&dec); return -1; }
        aji_frame in_f, out_f;
        fill_aji_frame(c, &in_f, dec, 0);
        fill_aji_frame(c, &out_f, cur, 0);
        int ret = aji_resize(c->aji, &in_f, &out_f, c->stream);
        if (ret != AJI_OK)
            loge("aji_resize: %d: %s", ret, aji_last_error(c->aji));
        int r = ret == AJI_OK ? sync_aji(c) : -1;
        av_frame_free(&dec);
        if (r < 0) { av_frame_free(&cur); return -1; }
    }
    if (!c->src_prev) {
        c->src_prev = cur;
        return 0;
    }

    AVFrame *base = upscale_one(c, c->src_prev);
    int rc = base ? emit_output(c, base) : -1;
    int phases = c->rnum / c->rden;        /* integer factor: F-1 phases */
    for (int k = 1; rc == 0 && k < phases; k++) {
        AVFrame *interp = alloc_work(c);
        if (!interp) { rc = -1; break; }
        aji_frame a, b, o;
        fill_aji_frame(c, &a, c->src_prev, 0);
        fill_aji_frame(c, &b, cur, 0);
        fill_aji_frame(c, &o, interp, 0);
        int ret = aji_infer_rife(c->aji, &a, &b, (double)k / phases, &o,
                                 c->stream);
        if (ret == AJI_SCENE) {
            c->scene_dupes++;
            rc = emit_output(c, base);
        } else if (ret != AJI_OK) {
            loge("aji_infer_rife: %d: %s", ret, aji_last_error(c->aji));
            rc = -1;
        } else {
            AVFrame *up = upscale_one(c, interp);
            rc = up ? emit_output(c, up) : -1;
            av_frame_free(&up);
        }
        av_frame_free(&interp);
    }
    av_frame_free(&base);
    av_frame_free(&c->src_prev);
    c->src_prev = cur;
    return rc;
}

/* one decoded frame into the active pipeline (NULL flushes it) */
static int push_frame(enc_ctx *c, AVFrame *fr)
{
    if (fr && (!c->passthrough || c->rife_only) && !(fr = to_device(c, fr)))
        return -1;
    if (!c->rife) return push_upscale(c, fr);
    if (c->rife_only) return rife_step(c, fr);   /* interpolate the frames as-is */
    return c->rife_first ? push_rife_first(c, fr) : push_rife(c, fr);
}

/* ---- main demux/decode loop ---------------------------------------------- */

static int run(enc_ctx *c)
{
    AVPacket *pkt = av_packet_alloc();
    int rc = -1;

    while (av_read_frame(c->ifmt, pkt) >= 0) {
        if (pkt->stream_index == c->vstream) {
            AV(avcodec_send_packet(c->dec, pkt));
            for (;;) {
                AVFrame *fr = av_frame_alloc();
                int r = avcodec_receive_frame(c->dec, fr);
                if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) {
                    av_frame_free(&fr);
                    break;
                }
                if (r < 0) { av_frame_free(&fr); AV(r); }
                /* fr is owned by the pipeline now */
                if (push_frame(c, fr) < 0) goto done;
            }
        } else if (c->smap && c->smap[pkt->stream_index] >= 0) {
            AVStream *in = c->ifmt->streams[pkt->stream_index];
            int oidx = c->smap[pkt->stream_index];
            AVPacket *out = av_packet_clone(pkt);
            out->stream_index = oidx;
            av_packet_rescale_ts(out, in->time_base,
                                 c->opened ? c->ofmt->streams[oidx]->time_base
                                           : in->time_base);
            if (c->opened) {
                av_interleaved_write_frame(c->ofmt, out);
                av_packet_free(&out);
            } else {
                /* header not written yet: buffer */
                if (c->aux_n == c->aux_cap) {
                    c->aux_cap = c->aux_cap ? c->aux_cap * 2 : 16;
                    c->aux_buf = av_realloc_f(c->aux_buf, c->aux_cap,
                                              sizeof(*c->aux_buf));
                }
                c->aux_buf[c->aux_n++] = out;
            }
        }
        av_packet_unref(pkt);
    }

    /* flush decoder */
    avcodec_send_packet(c->dec, NULL);
    for (;;) {
        AVFrame *fr = av_frame_alloc();
        int r = avcodec_receive_frame(c->dec, fr);
        if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) { av_frame_free(&fr); break; }
        if (r < 0) { av_frame_free(&fr); AV(r); }
        if (push_frame(c, fr) < 0) goto done;
    }

    /* flush pipeline */
    if (push_frame(c, NULL) < 0) goto done;

    if (!c->opened) { loge("no video frames produced"); goto done; }

    /* flush encoder + trailer */
    if (drain_encoder(c, 1) < 0) goto done;
    AV(av_write_trailer(c->ofmt));

    if (c->rife) {
        int64_t expect = (int64_t)(c->src_frames_est *
                         (double)c->rnum / c->rden + 0.5);
        loge("rife: produced %lld frames (~expected %lld), %lld scene dupes",
             (long long)c->out_frames, (long long)expect,
             (long long)c->scene_dupes);
    }
    progress("encode", c->out_frames, c->out_frames, 0, 100);
    rc = 0;
done:
fail:
    av_packet_free(&pkt);
    return rc;
}

/* ---- cleanup ------------------------------------------------------------- */

static void cleanup(enc_ctx *c)
{
    for (int i = 0; i < c->ring_count; i++) {
        slot *s = &c->ring[(c->ring_head - 1 - i + MAX_RING) % MAX_RING];
        av_frame_free(&s->out);
        av_frame_free(&s->in_ref);
    }
    av_frame_free(&c->up_prev);
    av_frame_free(&c->up_cur);
    av_frame_free(&c->up_interp);
    av_frame_free(&c->src_prev);
    if (c->in_sws) sws_freeContext(c->in_sws);
    av_frame_free(&c->host_frame);
    if (c->graph) avfilter_graph_free(&c->graph);
    if (c->sws) sws_freeContext(c->sws);
    for (int i = 0; i < c->aux_n; i++) av_packet_free(&c->aux_buf[i]);
    av_free(c->aux_buf);
    av_free(c->smap);
    if (c->aji) aji_destroy(&c->aji);
    if (c->stream) cudaStreamDestroy(c->stream);
    av_buffer_unref(&c->aji_pool);
    av_buffer_unref(&c->src_pool);
    av_buffer_unref(&c->work_pool);
    if (c->enc) avcodec_free_context(&c->enc);
    if (c->dec) avcodec_free_context(&c->dec);
    if (c->ofmt) {
        if (c->ofmt->pb && !(c->ofmt->oformat->flags & AVFMT_NOFILE))
            avio_closep(&c->ofmt->pb);
        avformat_free_context(c->ofmt);
    }
    if (c->ifmt) avformat_close_input(&c->ifmt);
    av_buffer_unref(&c->hw_device);
    if (c->ctx_pushed) { CUcontext old; cuCtxPopCurrent(&old); }
}

/* ---- CLI ----------------------------------------------------------------- */

static void usage(void)
{
    fprintf(stderr,
"aji_encode --input <f> --output <f> --conf <c> --slot <N> "
"--model-dir <d> [--rife-model-dir <d>] [--trtexec <p>]\n"
"           [--vcodec hevc_nvenc|h264_nvenc|hevc_amf|hevc_qsv|libx265|libx264|ffv1]"
" [--vquality \"<args>\"]\n"
"           [--pix-fmt yuv420p|yuv420p10|yuv444p|yuv444p10]"
" (default yuv420p10; 4:4:4 = software encoder)\n"
"           [--final-resize-height H | --final-resize-factor PCT]"
" [--overwrite]\n"
"           [--no-audio] [--no-subs] [--no-chapters]"
" [--progress none|line|json]\n"
"           [--build-only] [--log <f>] [--decoder auto|nvdec|cpu]"
" [--backend tensorrt|directml]\n"
"           (direct mode: --engine <e> --max-width W --max-height H)\n");
}

int main(int argc, char **argv)
{
    enc_ctx c = {0};
    c.o.slot = 1;
    c.o.vcodec = "hevc_nvenc";
    c.o.backend = "tensorrt";
    c.o.progress = PROG_LINE;
    c.o.decoder = DEC_AUTO;
    c.o.pipeline_depth = 4;
    c.o.pix_fmt = OUT_420P10;     /* default: 10-bit 4:2:0 (matches old pipeline) */
    g_log = stderr;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--overwrite")) c.o.overwrite = 1;
        else if (!strcmp(a, "--no-audio")) c.o.no_audio = 1;
        else if (!strcmp(a, "--no-subs")) c.o.no_subs = 1;
        else if (!strcmp(a, "--no-chapters")) c.o.no_chapters = 1;
        else if (!strcmp(a, "--build-only")) c.o.build_only = 1;
        else if (!strcmp(a, "--no-zerocopy")) c.o.no_zerocopy = 1;
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(); return 0; }
        else if (i >= argc - 1) { usage(); return 2; }
        else if (!strcmp(a, "--input")) c.o.input = argv[++i];
        else if (!strcmp(a, "--output")) c.o.output = argv[++i];
        else if (!strcmp(a, "--conf")) c.o.conf = argv[++i];
        else if (!strcmp(a, "--slot")) c.o.slot = atoi(argv[++i]);
        else if (!strcmp(a, "--model-dir")) c.o.model_dir = argv[++i];
        else if (!strcmp(a, "--rife-model-dir")) c.o.rife_model_dir = argv[++i];
        else if (!strcmp(a, "--trtexec")) c.o.trtexec = argv[++i];
        else if (!strcmp(a, "--trtexec-env")) c.o.trtexec_env = argv[++i];
        else if (!strcmp(a, "--backend")) c.o.backend = argv[++i];
        else if (!strcmp(a, "--vcodec")) c.o.vcodec = argv[++i];
        else if (!strcmp(a, "--vquality")) c.o.vquality = argv[++i];
        else if (!strcmp(a, "--pix-fmt")) {
            const char *v = argv[++i];
            if (!strcmp(v, "yuv420p"))        c.o.pix_fmt = OUT_420P8;
            else if (!strcmp(v, "yuv420p10")) c.o.pix_fmt = OUT_420P10;
            else if (!strcmp(v, "yuv444p"))   c.o.pix_fmt = OUT_444P8;
            else if (!strcmp(v, "yuv444p10")) c.o.pix_fmt = OUT_444P10;
            else { loge("unknown --pix-fmt '%s' (yuv420p|yuv420p10|yuv444p|"
                        "yuv444p10)", v); return 2; }
        }
        else if (!strcmp(a, "--final-resize-height")) c.o.final_h = atoi(argv[++i]);
        else if (!strcmp(a, "--final-resize-factor")) c.o.final_pct = atoi(argv[++i]);
        else if (!strcmp(a, "--progress")) {
            const char *v = argv[++i];
            c.o.progress = !strcmp(v, "none") ? PROG_NONE :
                           !strcmp(v, "json") ? PROG_JSON : PROG_LINE;
        }
        else if (!strcmp(a, "--log")) c.o.log_path = argv[++i];
        else if (!strcmp(a, "--decoder")) {
            const char *v = argv[++i];
            c.o.decoder = !strcmp(v, "nvdec") ? DEC_NVDEC :
                          !strcmp(v, "cpu") ? DEC_CPU : DEC_AUTO;
        }
        else if (!strcmp(a, "--engine")) c.o.engine = argv[++i];
        else if (!strcmp(a, "--max-width")) c.o.max_w = atoi(argv[++i]);
        else if (!strcmp(a, "--max-height")) c.o.max_h = atoi(argv[++i]);
        else if (!strcmp(a, "--pipeline-depth")) c.o.pipeline_depth = atoi(argv[++i]);
        else { loge("unknown arg: %s", a); usage(); return 2; }
    }

    g_progress = c.o.progress;
    if (c.o.log_path) {
        g_log = fopen(c.o.log_path, "w");
        if (!g_log) { perror(c.o.log_path); return 1; }
    }
    av_log_set_callback(av_log_cb);
    av_log_set_level(AV_LOG_WARNING);

    /* Must agree with the conf's [global] backend, which is what libaji's
     * dispatcher loads; this only picks the device the frames live on. */
    if (!av_strcasecmp(c.o.backend, "directml")) {
        c.dml = 1;
    } else if (av_strcasecmp(c.o.backend, "tensorrt")) {
        loge("unknown --backend '%s' (tensorrt|directml)", c.o.backend);
        return 2;
    }
    if (c.dml && c.o.pix_fmt >= OUT_444P8) {
        loge("the DirectML backend outputs 4:2:0 only (--pix-fmt yuv420p or "
             "yuv420p10)");
        return 2;
    }
    if (c.dml && c.o.engine) {
        loge("direct mode (--engine) is TensorRT-only");
        return 2;
    }
    if (!c.o.input || (!c.o.build_only && !c.o.output)) {
        usage(); return 2;
    }
    if (!c.o.conf && !c.o.engine) { loge("need --conf or --engine"); return 2; }
    if (c.o.final_h > 0 && c.o.final_pct > 0) {
        loge("--final-resize-height and --final-resize-factor are exclusive");
        return 2;
    }
    if (c.o.pix_fmt >= OUT_444P8 && is_nvenc(c.o.vcodec)) {
        loge("4:4:4 output needs a software encoder (libx265 main444 / "
             "libx264 high444); %s 4:4:4 not supported in v1", c.o.vcodec);
        return 2;
    }
    c.ring_depth = c.o.pipeline_depth;
    if (c.ring_depth < 1) c.ring_depth = 1;
    if (c.ring_depth > MAX_RING - 1) c.ring_depth = MAX_RING - 1;

    int rc = 1;
    if (open_input(&c) < 0) goto out;
    resolve_pixfmt(&c);    /* needs src_aji_fmt from open_input */
    if ((c.dml ? setup_d3d11(&c) : setup_cuda(&c)) < 0) goto out;
    if (init_aji(&c) < 0) goto out;

    if (c.o.build_only) {
        loge("build-only: engines ready, exiting");
        rc = 0;
        goto out;
    }

    if (resize_needed(&c)) compute_final_dims(&c);
    /* RIFE-only chains interpolate into this pool too (at the source dims) */
    if ((!c.passthrough || c.rife_only) && init_aji_pool(&c) < 0) goto out;
    if (open_decoder(&c) < 0) goto out;
    if (open_output(&c) < 0) goto out;
    if (run(&c) < 0) goto out;
    rc = 0;

out:
    cleanup(&c);
    if (g_log && g_log != stderr) fclose(g_log);
    return rc;
}
