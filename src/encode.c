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
 * --deinterlace runs the decoded frames through a libavfilter graph first
 * (IVTC, or bwdif at field rate); --preview keeps a JPEG of the latest
 * output frame up to date for a GUI, and --preview-session serves a GUI's
 * player instead of encoding (frames at any position, or played from it).
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

#include <limits.h>
#include <math.h>
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
#include <libavutil/dovi_meta.h>
#include <libavutil/mastering_display_metadata.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#ifdef _WIN32
#define COBJMACROS      /* C-style DXGI calls (GPU listing, amf_pick_adapter) */
#include <windows.h>
#include <dxgi1_6.h>
#include <d3d12.h>
#include <fcntl.h>
#include <io.h>
#include <libavutil/hwcontext_d3d11va.h>
#else
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#endif
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/time.h>
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
enum deint_mode { DEINT_OFF, DEINT_IVTC, DEINT_BWDIF };
enum square_mode { SQUARE_OFF, SQUARE_BEFORE, SQUARE_AFTER };
enum damage_mode { DAMAGE_FAIL, DAMAGE_CONTINUE };

/* exit status: 1 = error (a read error included), 2 = bad arguments,
 * 3 = damaged input with --on-damage fail */
#define EXIT_DAMAGED 3

/* output chroma subsampling + bit depth, chosen via --pix-fmt and decoupled
 * from the source. 16-bit 4:4:4 is libaji's intermediate, never the file. */
enum out_pixfmt { OUT_420P8, OUT_420P10, OUT_444P8, OUT_444P10 };

typedef struct {
    const char *input, *output;
    const char *conf, *model_dir, *rife_model_dir;
    const char *trtexec, *trtexec_env;
    int slot;
    const char *backend;
    uint64_t device;        /* --device: a GPU id from --probe-encoders; 0 = default */
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
    enum deint_mode deint;
    enum square_mode square; /* --square-pixels */
    const char *preview;    /* --preview: JPEG of the latest output frame */
    int preview_w;
    int session;            /* --preview-session: serve a player, no encode */
    enum damage_mode on_damage;

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

/* Errors logged by the video decoder and by the demuxer while run() decodes.
 * Some damage reaches the API only as a log line: the HEVC decoder skips a
 * slice it can't parse, or a frame whose references are gone ("Error
 * constructing the frame RPS"), and Matroska resyncs past a broken element,
 * all without an error code or a flagged frame. The decoder's frame threads
 * log too, hence the atomic counters. */
#ifdef _WIN32
typedef volatile LONG err_count;
#define ERR_INC(x) InterlockedIncrement(&(x))
#define ERR_GET(x) InterlockedCompareExchange(&(x), 0, 0)
#else
typedef long err_count;
#define ERR_INC(x) __atomic_add_fetch(&(x), 1, __ATOMIC_RELAXED)
#define ERR_GET(x) __atomic_load_n(&(x), __ATOMIC_RELAXED)
#endif
static err_count g_dec_errs, g_demux_errs;
static int g_count_errs;        /* set while run() decodes */

static void av_log_cb(void *p, int level, const char *fmt, va_list ap)
{
    if (level <= AV_LOG_ERROR && g_count_errs && p) {
        const AVClass *cls = *(const AVClass **)p;
        if (cls == avcodec_get_class()) {
            const AVCodecContext *ac = p;
            if (ac->codec_type == AVMEDIA_TYPE_VIDEO && ac->codec &&
                av_codec_is_decoder(ac->codec))
                ERR_INC(g_dec_errs);
        } else if (cls == avformat_get_class() &&
                   ((const AVFormatContext *)p)->iformat) {
            ERR_INC(g_demux_errs);
        }
    }
    if (level > av_log_get_level())
        return;
    /* --deinterlace ivtc: fieldmatch warns about every frame it leaves combed
     * (bwdif then deinterlaces it, as intended), which would flood the log of
     * a whole DVD */
    const AVClass *cls = p ? *(const AVClass **)p : NULL;
    if (cls && level >= AV_LOG_WARNING && cls->item_name &&
        strstr(cls->item_name(p), "fieldmatch"))
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

/* Fill in what an untagged YUV source leaves out, the way mpv does
 * (mp_image_params_guess_csp, which runs on every decoded frame before
 * vf_animejanai sees it), from the source's frame size w x h:
 *   matrix     BT.709 if w >= 1280 or h > 576, else BT.601
 *              (mp_csp_guess_colorspace); tagged bt470bg at 576 lines and
 *              smpte170m otherwise, the same matrix under its PAL and NTSC
 *              names
 *   range      limited
 *   primaries  BT.2020 for a BT.2020 matrix, BT.709 for BT.709; otherwise
 *              by size (mp_csp_guess_primaries): BT.709 for HD, bt470bg
 *              (625-line) at 576 lines, smpte170m (525-line) at 480/486,
 *              BT.709 for any other size
 *   transfer   BT.1886 for limited range (tagged bt709), sRGB for full
 * Only missing fields are filled, so a tagged source keeps its tags.
 * libaji converts with the matrix and range and writes its output in
 * them, and the encoder is tagged with all four: the upscaled file is
 * HD-sized, and a player guessing on its own would take BT.709. */
static void guess_colorimetry(int *spc, int *range, int *pri, int *trc,
                              int w, int h)
{
    const int hd = w >= 1280 || h > 576;
    if (*spc == AVCOL_SPC_UNSPECIFIED || *spc == AVCOL_SPC_RESERVED)
        *spc = hd ? AVCOL_SPC_BT709 : h == 576 ? AVCOL_SPC_BT470BG
                                               : AVCOL_SPC_SMPTE170M;
    if (*range == AVCOL_RANGE_UNSPECIFIED)
        *range = AVCOL_RANGE_MPEG;
    if (*pri == AVCOL_PRI_UNSPECIFIED || *pri == AVCOL_PRI_RESERVED ||
        *pri == AVCOL_PRI_RESERVED0) {
        if (*spc == AVCOL_SPC_BT2020_NCL || *spc == AVCOL_SPC_BT2020_CL)
            *pri = AVCOL_PRI_BT2020;
        else if (*spc == AVCOL_SPC_BT709 || hd)
            *pri = AVCOL_PRI_BT709;
        else
            *pri = h == 576 ? AVCOL_PRI_BT470BG
                 : h == 480 || h == 486 ? AVCOL_PRI_SMPTE170M
                 : AVCOL_PRI_BT709;
    }
    if (*trc == AVCOL_TRC_UNSPECIFIED || *trc == AVCOL_TRC_RESERVED ||
        *trc == AVCOL_TRC_RESERVED0)
        *trc = *range == AVCOL_RANGE_JPEG ? AVCOL_TRC_IEC61966_2_1
                                          : AVCOL_TRC_BT709;
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
#define MAX_HELD 256    /* video packets kept until NVDEC is known to work */

typedef struct {
    AVFrame  *out;       /* aji output (or decoded, passthrough) CUDA frame */
    AVFrame  *in_ref;    /* decoded input kept alive across the async infer */
    uint64_t  ticket;
    int64_t   pts;       /* output PTS in the video stream time_base */
} slot;

/* a still-image encoder (scaler + PNG/MJPEG), kept across frames */
typedef struct {
    struct SwsContext *sws;
    AVCodecContext    *enc;
} img_enc;

/* --preview-session: an encoded source frame, sent with its output */
typedef struct {
    int64_t   pts;
    AVPacket *pkt;
} src_image;

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
    int              dec_sw;         /* get_hw_format fell back to software */
    int              dec_hw;         /* get_hw_format picked CUDA */
    int              dec_hold;       /* keeping packets for a CPU decoder */
    AVPacket        *dec_held[MAX_HELD];
    int              dec_held_n;
    AVFrame         *dec_pend[MAX_HELD];  /* frames it put out while being
                                             given them again */
    int              dec_pend_head, dec_pend_n;
    int              dml;            /* DirectML backend: frames are shared D3D11
                                        textures (libav D3D11 frames), and no
                                        CUDA call is ever made */
    void            *d3d11_dev;      /* ID3D11Device* of the D3D11VA hwdevice */
    int              gpu_index;      /* DirectML: the upscaling GPU's DXGI index, -1 if unknown */
    unsigned         gpu_vendor;     /* ... and its PCI vendor id (AMF/QSV follow it) */

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
    int             rnum, rden;      /* its factor, reduced (rnum >= rden) */
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
    AVRational       out_tb;         /* video encoder time_base: the source
                                        stream's, divided by the RIFE factor so
                                        interpolated frames land on whole ticks */
    AVRational       out_fps;        /* nominal (encoder rate control, tags) */
    AVRational       src_tb;         /* video stream time_base (out_tb's base) */
    AVRational       in_tb;          /* time base of the frames entering the
                                        pipeline (= src_tb, or the
                                        deinterlacer's output) */
    AVRational       src_fps;        /* source frame rate (guessed) */
    int64_t          start_us;       /* input start, subtracted from every
                                        stream's timestamps (AV_TIME_BASE) */
    int64_t          last_pts;       /* last frame timestamp (out_tb) */
    int64_t          frame_dur;      /* nominal frame duration (out_tb) */
    int64_t          seq_pts[64];    /* temporal: timestamp by sequence number */
    int             *aux_src;        /* input stream of each buffered aux packet */
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
    int      rife_acc;               /* where the next output falls past the
                                        held frame (up_prev / src_prev), in
                                        1/rnum of the interval to its
                                        successor: 0 = on the held frame */
    AVFrame *src_prev;               /* RIFE-first: previous work-res source */

    /* bookkeeping */
    int64_t  out_frames;
    /* PROGRESS fps=: (time, out_frames) at each progress line, the oldest
     * first; fps_t0 is when the first frame went to the encoder */
    int64_t  fps_t[64], fps_n[64];
    int      fps_head, fps_count;
    int64_t  fps_t0;
    int64_t  src_frames_est;
    int64_t  scene_dupes;
    int      color_space, color_range, color_pri, color_trc, chroma_loc;
    int      src_rgb;                /* RGB source, converted to YUV on upload */
    int      pad_r, pad_b;           /* odd source: padded by one column/row
                                        for the engine (src_w/h are padded) */
    int      crop_w, crop_h;         /* upscaled size cropped back to the
                                        source's true geometry */
    AVRational src_sar;              /* source sample aspect ratio */
    AVRational out_sar;              /* output's: the source's, or 1:1 once
                                        --square-pixels has stretched it */
    int      sq_before;              /* --square-pixels before: frames are
                                        stretched to src_w x src_h on upload */
    int      sq_after;               /* --square-pixels after: the final
                                        resize stretches them */

    /* --deinterlace: decoded (host) frames -> filter graph -> pipeline */
    AVFilterGraph   *dgraph;
    AVFilterContext *dsrc, *dsink;
    AVRational       dec_tb;         /* decoded frames' time base (stream's) */
    AVRational       dec_fps;
    int              dg_fmt, dg_spc, dg_range;  /* what the graph was built for */
    int              dg_fed;         /* a frame went in (no rebuilding after) */

    /* --preview */
    int64_t          preview_at;     /* av_gettime_relative() of the last write */
    img_enc          preview_img;

    /* --preview-session: the request being served */
    uint64_t         sess_cmd;       /* its command number (cmd_wait) */
    long long        sess_id;        /* the GUI's id for it, echoed back */
    int              sess_play;      /* play from the position, else one frame */
    int              sess_w;         /* image width, 0 = full size */
    int64_t          sess_at;        /* the position (out_tb) */
    int64_t          sess_target;    /* first frame at or after it, or NOPTS */
    int              sess_after;     /* frames pushed since the target */
    int              sess_done;      /* one frame: sent */
    int              sess_abort;     /* a newer command came in */
    int64_t          sess_clock;     /* play: wall clock (us) of sess_pts0 */
    int64_t          sess_pts0;
    img_enc          sess_src_img;
    src_image        sess_src[16];   /* source images awaiting their output */
    int              sess_src_n;

    /* damaged input (see "damaged input" below) */
    int64_t          dec_frames;     /* frames the decoder has put out: the
                                        source index of the next one */
    int64_t          vstart_us;      /* the video stream's start (AV_TIME_BASE) */
    double           pkt_t;          /* time of the last video packet sent (s) */
    double           frame_t;        /* time of the last decoded frame (s) */
    long             dec_errs_seen, demux_errs_seen;
    int              pend;           /* damage found in a packet, waiting to
                                        be tied to a flagged frame */
    double           pend_t;
    int              pend_wait;      /* frames left to wait for one */
    int              spot_open;      /* a damaged spot is being reported */
    double           spot_first, spot_last;
    int64_t          damage_spots;
    int              damage_stop;    /* --on-damage fail: stop now */

    /* HDR10 static metadata (mastering display, content light level) from
     * the first decoded frame, for the encoder and the container */
    AVFrameSideData **hdr_sd;
    int              nb_hdr_sd;
    int              hdr_seen;
} enc_ctx;

/* forward decls */
static int emit_output(enc_ctx *c, AVFrame *cuda_out, int64_t pts);
static int drain_encoder(enc_ctx *c, int flush);

/* ---- decoder: force NVDEC's CUDA output format --------------------------- */

static enum AVPixelFormat g_hw_pixfmt = AV_PIX_FMT_CUDA;

static enum AVPixelFormat get_hw_format(AVCodecContext *ctx,
                                        const enum AVPixelFormat *fmts)
{
    enc_ctx *c = ctx->opaque;
    for (const enum AVPixelFormat *p = fmts; *p != AV_PIX_FMT_NONE; p++)
        if (*p == g_hw_pixfmt) {
            if (c) c->dec_hw = 1;
            return *p;
        }
    /* NVDEC can't take this stream (libav drops cuda from the list once its
     * hwaccel init fails, e.g. 4:4:4 H.264): --decoder auto carries on in
     * software, and to_device uploads the frames */
    if (c && c->o.decoder == DEC_AUTO) {
        for (const enum AVPixelFormat *p = fmts; *p != AV_PIX_FMT_NONE; p++) {
            const AVPixFmtDescriptor *d = av_pix_fmt_desc_get(*p);
            if (d && !(d->flags & AV_PIX_FMT_FLAG_HWACCEL)) {
                loge("NVDEC can't decode this stream; decoding on the CPU");
                c->dec_sw = 1;
                return *p;
            }
        }
    }
    return AV_PIX_FMT_NONE;
}

/* ---- GPUs ---------------------------------------------------------------- */

/* --device takes the id the image ABI gives a GPU (aji_img_device.id): its PCI
 * location, (1 << 63) | domain << 32 | bus << 8 | device, so a card keeps its
 * id across reboots and has the same one on both backends. DirectML falls
 * back to the adapter LUID (bit 63 clear) if Windows reports no address. */
#define GPU_ID_PCI    (1ull << 63)
#define GPU_ID_DOMAIN 0x7fffffff00000000ull

/* D3DKMT reports no PCI domain, so a DirectML id has domain 0: PCI ids
 * compare without it unless both carry one. */
static int same_gpu(uint64_t a, uint64_t b)
{
    if ((a & GPU_ID_PCI) && (b & GPU_ID_PCI) &&
        (!(a & GPU_ID_DOMAIN) || !(b & GPU_ID_DOMAIN))) {
        a &= ~GPU_ID_DOMAIN;
        b &= ~GPU_ID_DOMAIN;
    }
    return a == b;
}

/* A CUDA device's id, as aji_img_trt.cpp's pci_id computes it. */
static uint64_t cuda_device_id(CUdevice d)
{
    int dom = 0, bus = 0, dev = 0;
    cuDeviceGetAttribute(&dom, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, d);
    cuDeviceGetAttribute(&bus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, d);
    cuDeviceGetAttribute(&dev, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, d);
    return GPU_ID_PCI | ((uint64_t)(uint32_t)dom << 32) |
           ((uint64_t)(uint32_t)bus << 8) | (uint64_t)(dev & 0xff);
}

/* The CUDA device ordinal (libav's CUDA device string) of the GPU with id
 * `want`, or -1. */
static int cuda_ordinal(uint64_t want)
{
    int n = 0;
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGetCount(&n) != CUDA_SUCCESS)
        return -1;
    for (int i = 0; i < n; i++) {
        CUdevice d;
        if (cuDeviceGet(&d, i) == CUDA_SUCCESS && same_gpu(want, cuda_device_id(d)))
            return i;
    }
    return -1;
}

#ifdef _WIN32
/* D3DKMTQueryAdapterInfo for the adapter `luid`; 0 if anything fails. The
 * D3DKMT structs mirror d3dkmthk.h and the calls resolve from gdi32 at
 * runtime. Same helpers as aji_img_dml.cpp. */
static int kmt_query(LUID luid, UINT type, void *data, UINT size)
{
    typedef struct { LUID luid; UINT handle; } OpenFromLuid;
    typedef struct { UINT handle; UINT type; void *data; UINT size; } QueryInfo;
    typedef struct { UINT handle; } CloseAdapter;
    typedef LONG(APIENTRY *OpenFn)(OpenFromLuid *);
    typedef LONG(APIENTRY *QueryFn)(const QueryInfo *);
    typedef LONG(APIENTRY *CloseFn)(const CloseAdapter *);
    HMODULE gdi = GetModuleHandleW(L"gdi32.dll");
    if (!gdi) gdi = LoadLibraryW(L"gdi32.dll");
    if (!gdi) return 0;
    OpenFn  open  = (OpenFn)(void *)GetProcAddress(gdi, "D3DKMTOpenAdapterFromLuid");
    QueryFn query = (QueryFn)(void *)GetProcAddress(gdi, "D3DKMTQueryAdapterInfo");
    CloseFn close = (CloseFn)(void *)GetProcAddress(gdi, "D3DKMTCloseAdapter");
    if (!open || !query || !close) return 0;
    OpenFromLuid o = {luid, 0};
    if (open(&o) != 0) return 0;
    QueryInfo q = {o.handle, type, data, size};
    int ok = query(&q) == 0;
    CloseAdapter cl = {o.handle};
    close(&cl);
    return ok;
}

/* An indirect-display adapter: a virtual monitor (Parsec, Meta Quest Link,
 * any IddCx driver). DXGI lists one per virtual display, each describing the
 * GPU that renders for it, so one card would show up several times. The
 * kernel adapter type tells them apart; a failed lookup keeps the adapter. */
static int is_indirect_display(LUID luid)
{
    UINT type = 0;
    return kmt_query(luid, 15 /* KMTQAITYPE_ADAPTERTYPE */, &type, sizeof type) &&
           (type & (1u << 6) /* IndirectDisplayDevice */);
}

/* An adapter's id, as aji_img_dml.cpp's device_id computes it. */
static uint64_t adapter_id(LUID luid)
{
    struct { UINT bus, device, function; } addr = {0, 0, 0}; /* D3DKMT_ADAPTERADDRESS */
    if (kmt_query(luid, 6 /* KMTQAITYPE_ADAPTERADDRESS */, &addr, sizeof addr))
        return GPU_ID_PCI | ((uint64_t)addr.bus << 8) | (uint64_t)(addr.device & 0xff);
    return ((uint64_t)(uint32_t)luid.HighPart << 32) | (uint64_t)luid.LowPart;
}

typedef struct {
    int  index;                 /* DXGI adapter index (libav D3D11VA device string) */
    UINT vendor;                /* PCI vendor id */
    uint64_t id;                /* what --device takes */
    LUID luid;
    char name[128];             /* UTF-8 */
} gpu_adapter;

/* Hardware GPUs in DXGI order, without software adapters or the
 * virtual-display duplicates. Returns the count (at most max). */
static int list_gpus(gpu_adapter *out, int max)
{
    IDXGIFactory1 *f = NULL;
    if (FAILED(CreateDXGIFactory1(&IID_IDXGIFactory1, (void **)&f)))
        return 0;
    int n = 0;
    IDXGIAdapter1 *a = NULL;
    for (UINT i = 0; n < max &&
         IDXGIFactory1_EnumAdapters1(f, i, &a) != DXGI_ERROR_NOT_FOUND; i++) {
        DXGI_ADAPTER_DESC1 d;
        if (SUCCEEDED(IDXGIAdapter1_GetDesc1(a, &d)) &&
            !(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
            !is_indirect_display(d.AdapterLuid)) {
            out[n].index  = (int)i;
            out[n].vendor = d.VendorId;
            out[n].id     = adapter_id(d.AdapterLuid);
            out[n].luid   = d.AdapterLuid;
            WideCharToMultiByte(CP_UTF8, 0, d.Description, -1, out[n].name,
                                sizeof out[n].name, NULL, NULL);
            out[n].name[sizeof out[n].name - 1] = 0;
            n++;
        }
        IDXGIAdapter1_Release(a);
    }
    IDXGIFactory1_Release(f);
    return n;
}

/* Whether the adapter `luid` is integrated graphics: Direct3D 12 reports a
 * unified memory architecture (UMA) for GPUs that share system memory. -1 if
 * it can't tell (no d3d12.dll, or no Direct3D 12 device on the adapter).
 * d3d12.dll loads at runtime, so aji_encode doesn't depend on it. */
static int gpu_integrated(LUID luid)
{
    typedef HRESULT(WINAPI *CreateFn)(IUnknown *, D3D_FEATURE_LEVEL, REFIID, void **);
    static CreateFn create;
    static int loaded;
    if (!loaded) {
        loaded = 1;
        HMODULE m = LoadLibraryW(L"d3d12.dll");
        if (m) create = (CreateFn)(void *)GetProcAddress(m, "D3D12CreateDevice");
    }
    IDXGIFactory4 *f = NULL;
    if (!create || FAILED(CreateDXGIFactory1(&IID_IDXGIFactory4, (void **)&f)))
        return -1;
    int r = -1;
    IDXGIAdapter1 *a = NULL;
    ID3D12Device *dev = NULL;
    if (SUCCEEDED(IDXGIFactory4_EnumAdapterByLuid(f, luid, &IID_IDXGIAdapter1, (void **)&a)) &&
        SUCCEEDED(create((IUnknown *)a, D3D_FEATURE_LEVEL_11_0, &IID_ID3D12Device, (void **)&dev))) {
        D3D12_FEATURE_DATA_ARCHITECTURE arch = {0};
        if (SUCCEEDED(ID3D12Device_CheckFeatureSupport(dev, D3D12_FEATURE_ARCHITECTURE,
                                                       &arch, sizeof arch)))
            r = arch.UMA != 0;
        ID3D12Device_Release(dev);
    }
    if (a) IDXGIAdapter1_Release(a);
    IDXGIFactory4_Release(f);
    return r;
}

static const char *vendor_name(UINT v)
{
    return v == 0x10de ? "nvidia" : v == 0x1002 ? "amd" :
           v == 0x8086 ? "intel"  : "other";
}

/* The GPU --backend directml runs on without --device, as an index into
 * g[]: the first one Windows ranks for high performance, so the dedicated
 * GPU. DXGI adapter 0 is the GPU driving the main display, which on a laptop,
 * or a desktop with its monitor on the motherboard, is the integrated one.
 * 0 (DXGI order) if Windows has no ranking (before Windows 10 1803) or it
 * names none of g[]. */
static int default_gpu(const gpu_adapter *g, int n)
{
    IDXGIFactory6 *f = NULL;
    if (n <= 1 || FAILED(CreateDXGIFactory1(&IID_IDXGIFactory6, (void **)&f)))
        return 0;
    int pick = -1;
    IDXGIAdapter1 *a = NULL;
    for (UINT i = 0; pick < 0 &&
         SUCCEEDED(IDXGIFactory6_EnumAdapterByGpuPreference(f, i,
             DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, &IID_IDXGIAdapter1,
             (void **)&a)); i++) {
        DXGI_ADAPTER_DESC1 d;
        if (SUCCEEDED(IDXGIAdapter1_GetDesc1(a, &d)))
            for (int k = 0; k < n && pick < 0; k++)
                if (g[k].luid.LowPart == d.AdapterLuid.LowPart &&
                    g[k].luid.HighPart == d.AdapterLuid.HighPart)
                    pick = k;
        IDXGIAdapter1_Release(a);
    }
    IDXGIFactory6_Release(f);
    return pick < 0 ? 0 : pick;
}
#endif

/* ---- CUDA hwdevice + shared context -------------------------------------- */

static int setup_cuda(enc_ctx *c)
{
    /* No --device: CUDA device 0, the fastest unless CUDA_DEVICE_ORDER says
     * otherwise. */
    char dev[16];
    const char *devp = NULL;
    if (c->o.device) {
        int ord = cuda_ordinal(c->o.device);
        if (ord < 0) {
            loge("--device %llu: no CUDA GPU has that id (aji_encode "
                 "--probe-encoders lists them)", (unsigned long long)c->o.device);
            return -1;
        }
        snprintf(dev, sizeof dev, "%d", ord);
        devp = dev;
    }
    if (av_hwdevice_ctx_create(&c->hw_device, AV_HWDEVICE_TYPE_CUDA,
                               devp, NULL, 0) < 0) {
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
 * frame pools, uploads and readbacks, on --device's GPU, else default_gpu's.
 * Decoding stays on the CPU: hardware decoder textures can't be shared to
 * D3D12. */
static int setup_d3d11(enc_ctx *c)
{
#ifdef _WIN32
    gpu_adapter g[16];
    int n = list_gpus(g, 16), k = -1;
    if (c->o.device) {
        for (int i = 0; i < n && k < 0; i++)
            if (same_gpu(g[i].id, c->o.device)) k = i;
        if (k < 0) {
            loge("--device %llu: no GPU has that id (aji_encode "
                 "--probe-encoders lists them)", (unsigned long long)c->o.device);
            return -1;
        }
    } else if (n > 0) {
        k = default_gpu(g, n);
    }
    char dev[16];
    if (k >= 0) {
        snprintf(dev, sizeof dev, "%d", g[k].index);
        c->gpu_index  = g[k].index;
        c->gpu_vendor = g[k].vendor;
    }
    if (av_hwdevice_ctx_create(&c->hw_device, AV_HWDEVICE_TYPE_D3D11VA,
                               k >= 0 ? dev : NULL, NULL, 0) < 0) {
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
    /* libaji needs even frame dims (4:2:0 chroma, and its kernels); odd
     * sources (853x480, ...) are padded with a replicated edge column/row
     * and the result cropped back */
    c->pad_r = c->src_w & 1;
    c->pad_b = c->src_h & 1;
    if (c->pad_r || c->pad_b) {
        loge("odd frame size %dx%d: padding to %dx%d for the engine",
             c->src_w, c->src_h, c->src_w + c->pad_r, c->src_h + c->pad_b);
        c->src_w += c->pad_r;
        c->src_h += c->pad_b;
    }

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
    /* RGB sources (gbrp, bgr0, ... lossless intermediates) are converted to
     * YUV on upload; make that BT.709 limited explicitly and tag the stream to
     * match, so libaji, the output tags and the conversion agree (swscale's
     * BT.601 default against BT.709-assuming tags shifted the colors) */
    c->src_rgb = d && (d->flags & AV_PIX_FMT_FLAG_RGB);
    if (c->src_rgb) {
        c->color_space = AVCOL_SPC_BT709;
        c->color_range = AVCOL_RANGE_MPEG;
        c->chroma_loc  = AVCHROMA_LOC_LEFT;
    } else {
        /* untagged: an SD source is BT.601, as in mpv, not BT.709 */
        const int spc0 = c->color_space;
        guess_colorimetry(&c->color_space, &c->color_range, &c->color_pri,
                          &c->color_trc, par->width, par->height);
        if (c->color_space != spc0)
            loge("untagged colour matrix: assuming %s for %dx%d",
                 av_color_space_name(c->color_space), par->width,
                 par->height);
    }

    /* upscaling scales both axes alike, so the pixel aspect carries over */
    c->src_sar = av_guess_sample_aspect_ratio(c->ifmt, vst, NULL);
    c->out_sar = c->src_sar;

    /* --square-pixels: non-square pixels (DVD, DV) are stretched to square,
     * along the narrow axis so no source detail is dropped (16:9 720x480 ->
     * 854x480, 4:3 -> 720x540). Before processing, the stretched size is
     * the size everything downstream sees (chain selection, engines); after,
     * the final resize does it. */
    if (c->o.square && c->src_sar.num > 0 && c->src_sar.den > 0 &&
        c->src_sar.num != c->src_sar.den) {
        int qw = par->width, qh = par->height;
        if (c->src_sar.num > c->src_sar.den)
            qw = (int)av_rescale(qw, c->src_sar.num, c->src_sar.den);
        else
            qh = (int)av_rescale(qh, c->src_sar.den, c->src_sar.num);
        qw = (qw + 1) & ~1;
        qh = (qh + 1) & ~1;
        c->out_sar = (AVRational){1, 1};
        if (c->o.square == SQUARE_BEFORE) {
            c->sq_before = 1;
            c->src_w = qw;              /* even: no padding needed */
            c->src_h = qh;
            c->pad_r = c->pad_b = 0;
        } else {
            c->sq_after = 1;
        }
        loge("square pixels (%s processing): %dx%d SAR %d:%d -> %dx%d",
             c->sq_before ? "before" : "after", par->width, par->height,
             c->src_sar.num, c->src_sar.den, qw, qh);
    }

    c->out_fps = av_guess_frame_rate(c->ifmt, vst, NULL);
    if (c->out_fps.num == 0) c->out_fps = (AVRational){24000, 1001};
    /* Output frames keep the source frames' own timestamps (variable frame
     * rate sources stay in sync with their audio), all streams shifted by the
     * input's start time so the output starts at 0 */
    c->src_tb = vst->time_base;
    c->in_tb = c->src_tb;
    c->src_fps = c->out_fps;
    c->start_us = c->ifmt->start_time != AV_NOPTS_VALUE ? c->ifmt->start_time
                                                         : 0;
    c->last_pts = AV_NOPTS_VALUE;
    /* DAMAGE times count from the video stream's own start */
    c->vstart_us = vst->start_time != AV_NOPTS_VALUE
                 ? av_rescale_q(vst->start_time, vst->time_base, AV_TIME_BASE_Q)
                 : c->start_us;

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

static int is_nvenc(const char *name)
{
    return name && strstr(name, "nvenc") != NULL;
}

static int is_amf(const char *name)
{
    return name && strstr(name, "_amf") != NULL;
}

static int is_qsv(const char *name)
{
    return name && strstr(name, "_qsv") != NULL;
}

/* GPU encoders (NVENC, AMD AMF, Intel QSV): they take the semi-planar 4:2:0
 * host formats (nv12/p010le) and no 4:4:4 here. Only NVENC also takes the
 * engine's CUDA/D3D11 frames zero-copy. */
static int is_hw_encoder(const char *name)
{
    return is_nvenc(name) || is_amf(name) || is_qsv(name);
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
    int hwenc = is_hw_encoder(c->o.vcodec);
    switch (c->o.pix_fmt) {
    case OUT_420P8:  c->ten_bit=0; c->is_444=0; break;
    case OUT_420P10: c->ten_bit=1; c->is_444=0; break;
    case OUT_444P8:  c->ten_bit=0; c->is_444=1; break;
    case OUT_444P10: c->ten_bit=1; c->is_444=1; break;
    }
    /* Encoder-input (and swscale-target) format. The GPU encoders accept
     * only the semi-planar 4:2:0 host formats (nv12/p010le), not planar
     * yuv420p10le (QSV not even yuv420p); software encoders take the planar
     * forms. 4:4:4 is software-only. */
    if (c->is_444)
        c->out_pixfmt = c->ten_bit ? AV_PIX_FMT_YUV444P10LE : AV_PIX_FMT_YUV444P;
    else if (hwenc)
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

static AVCodecContext *alloc_decoder(enc_ctx *c)
{
    const AVStream *vst = c->ifmt->streams[c->vstream];
    const AVCodec *codec = avcodec_find_decoder(vst->codecpar->codec_id);
    if (!codec) { loge("no decoder for video stream"); return NULL; }
    AVCodecContext *dec = avcodec_alloc_context3(codec);
    if (!dec) { loge("avcodec_alloc_context3 failed"); return NULL; }
    if (avcodec_parameters_to_context(dec, vst->codecpar) < 0) {
        avcodec_free_context(&dec);
        return NULL;
    }
    dec->pkt_timebase = vst->time_base;
    /* a frame carries its packet's opaque: DMG_TAG marks the packets the
     * demuxer flagged corrupt */
    dec->flags |= AV_CODEC_FLAG_COPY_OPAQUE;
    return dec;
}

/* Open c->dec as a CPU decoder. libavcodec defaults thread_count to 1 (the
 * ffmpeg CLI sets "auto" itself), which held H.264/HEVC/AV1 decode to one
 * core; take every core, by frame and by slice. Frame threading delays the
 * output by a few frames, which the decode loops drain at the end
 * (avcodec_send_packet(NULL)) and drop on a seek (avcodec_flush_buffers);
 * the frames and their timestamps are the same. NVDEC is left as it is. */
static int open_cpu_decoder(enc_ctx *c)
{
    avcodec_free_context(&c->dec);
    if (!(c->dec = alloc_decoder(c))) return -1;
    c->dec->thread_count = 0;
    c->dec->thread_type  = FF_THREAD_FRAME | FF_THREAD_SLICE;
    AV(avcodec_open2(c->dec, c->dec->codec, NULL));
    c->nvdec = 0;
    return 0;
fail:
    return -1;
}

/* libavcodec has an NVDEC hwaccel for this codec (some profiles of it) */
static int codec_has_cuda(const AVCodec *codec)
{
    for (int i = 0;; i++) {
        const AVCodecHWConfig *cfg = avcodec_get_hw_config(codec, i);
        if (!cfg) return 0;
        if (cfg->device_type == AV_HWDEVICE_TYPE_CUDA &&
            (cfg->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX))
            return 1;
    }
}

static int open_decoder(enc_ctx *c)
{
    /* deinterlacing filters run on the CPU: decode there too */
    const int want_hw = (c->o.decoder != DEC_CPU) && !c->o.no_zerocopy &&
                        !c->dml && !c->dgraph;
    if (!want_hw)
        return open_cpu_decoder(c);
    const AVCodec *codec =
        avcodec_find_decoder(c->ifmt->streams[c->vstream]->codecpar->codec_id);
    if (codec && !codec_has_cuda(codec) && c->o.decoder == DEC_AUTO) {
        loge("NVDEC can't decode %s; decoding on the CPU", codec->name);
        return open_cpu_decoder(c);
    }

    if (!(c->dec = alloc_decoder(c))) return -1;
    c->dec->hw_device_ctx = av_buffer_ref(c->hw_device);
    c->dec->get_format = get_hw_format;
    c->dec->opaque = c;
    int r = avcodec_open2(c->dec, c->dec->codec, NULL);
    if (r < 0 && c->o.decoder == DEC_AUTO) {
        loge("NVDEC open failed; falling back to CPU decode");
        return open_cpu_decoder(c);
    }
    if (r < 0) { AV(r); }
    c->nvdec = 1;
    /* --decoder auto: NVDEC may still turn the stream down once it sees
     * it (get_hw_format); keep the packets until then (dec_send) */
    c->dec_hold = c->o.decoder == DEC_AUTO;
    return 0;
fail:
    return -1;
}

static void dec_drop_held(enc_ctx *c)
{
    for (int i = 0; i < c->dec_held_n; i++)
        av_packet_free(&c->dec_held[i]);
    c->dec_held_n = 0;
}

static void dec_drop_pending(enc_ctx *c)
{
    while (c->dec_pend_n > 0) {
        av_frame_free(&c->dec_pend[c->dec_pend_head]);
        c->dec_pend_head = (c->dec_pend_head + 1) % MAX_HELD;
        c->dec_pend_n--;
    }
    c->dec_pend_head = 0;
}

/* avcodec_send_packet that takes the frames waiting to be received out
 * first (into dec_pend, for dec_receive) instead of returning EAGAIN */
static int dec_send_pending(enc_ctx *c, const AVPacket *pkt)
{
    int r;
    while ((r = avcodec_send_packet(c->dec, pkt)) == AVERROR(EAGAIN)) {
        AVFrame *fr = c->dec_pend_n < MAX_HELD ? av_frame_alloc() : NULL;
        if (!fr) return AVERROR(ENOMEM);
        if ((r = avcodec_receive_frame(c->dec, fr)) < 0) {
            av_frame_free(&fr);
            if (r == AVERROR_INVALIDDATA)
                continue;   /* before the first frame: not damage (run) */
            return r;
        }
        c->dec_pend[(c->dec_pend_head + c->dec_pend_n++) % MAX_HELD] = fr;
    }
    return r;
}

/* NVDEC turned the stream down after opening: libavcodec carries on in
 * software, but on one thread (threads can't be set on an open decoder).
 * Hand the stream to a threaded CPU decoder instead, giving it the packets
 * NVDEC was given; no frame has come out yet, so it starts from the same
 * point. */
static int dec_restart_cpu(enc_ctx *c, int eof)
{
    c->dec_hold = 0;
    int r = open_cpu_decoder(c) < 0 ? AVERROR(EINVAL) : 0;
    for (int i = 0; r >= 0 && i < c->dec_held_n; i++)
        r = dec_send_pending(c, c->dec_held[i]);
    dec_drop_held(c);
    if (r >= 0 && eof)
        r = dec_send_pending(c, NULL);
    return r;
}

/* avcodec_send_packet for the video decoder (NULL: end of stream). */
static int dec_send(enc_ctx *c, const AVPacket *pkt)
{
    if (c->dec_hold && pkt) {
        AVPacket *ref = c->dec_held_n < MAX_HELD ? av_packet_clone(pkt) : NULL;
        if (ref) {
            c->dec_held[c->dec_held_n++] = ref;
        } else {
            c->dec_hold = 0;        /* too many: NVDEC's fallback stays */
            dec_drop_held(c);
        }
    }
    const int r = avcodec_send_packet(c->dec, pkt);
    if (c->dec_hold && c->dec_sw)
        return dec_restart_cpu(c, !pkt);
    if (c->dec_hold && c->dec_hw) {
        c->dec_hold = 0;            /* NVDEC took it */
        dec_drop_held(c);
    }
    return r;
}

/* avcodec_receive_frame for the video decoder */
static int dec_receive(enc_ctx *c, AVFrame *fr)
{
    if (c->dec_pend_n > 0) {
        av_frame_move_ref(fr, c->dec_pend[c->dec_pend_head]);
        av_frame_free(&c->dec_pend[c->dec_pend_head]);
        c->dec_pend_head = (c->dec_pend_head + 1) % MAX_HELD;
        c->dec_pend_n--;
        return 0;
    }
    const int r = avcodec_receive_frame(c->dec, fr);
    if (r >= 0 && c->dec_hold) {
        c->dec_hold = 0;            /* a frame is out: too late to restart */
        dec_drop_held(c);
    }
    return r;
}

/* drop everything the decoder holds (a seek) */
static void dec_flush(enc_ctx *c)
{
    dec_drop_pending(c);
    dec_drop_held(c);
    avcodec_flush_buffers(c->dec);
}

/* ---- damaged input ------------------------------------------------------- *
 *
 * Damage is found the same way whatever the decoder (NVDEC or CPU) and the
 * container:
 *   - the decoder rejects data: avcodec_send_packet / avcodec_receive_frame
 *     return AVERROR_INVALIDDATA (the packet is dropped);
 *   - a decoded frame is flagged: decode_error_flags (concealed) or
 *     AV_FRAME_FLAG_CORRUPT;
 *   - the demuxer flags a video packet AV_PKT_FLAG_CORRUPT (MPEG-TS
 *     continuity errors, a sample cut short by the end of the file) or
 *     av_read_frame returns AVERROR_INVALIDDATA;
 *   - the video decoder or the demuxer logs an error (g_dec_errs).
 * Any other error (ENOMEM, a hardware/device failure, the encoder) stays
 * fatal. NVDEC decodes the macroblocks in hardware and libavcodec does not
 * ask it for its decode status, so macroblock-level damage it conceals is
 * not seen; what libavcodec parses itself (NAL units, parameter sets, slice
 * headers, references) is, as on the CPU.
 *
 * Errors before the decoder's first frame are not damage: a recording that
 * starts mid-GOP has undecodable packets until its first keyframe, and NVDEC
 * turning a stream down (--decoder auto) logs errors before the CPU decoder
 * takes over.
 *
 * The first three name a packet, not a frame, and with frame threads the
 * error surfaces a few packets late; such a find waits (pend) for the
 * decoder's next few frames, and if one of them is flagged it is reported
 * as that frame. Finds less than a second apart are one damaged spot, so a
 * broken slice (an error, its concealed frame, the frames left without
 * references) is one line. Each spot prints, on stdout,
 *   DAMAGE t=<s since the video stream's start, 3 decimals> frame=<n | -1>
 * with the first frame of the spot the decoder put out (its source index),
 * or -1 if the spot starts with a packet. --on-damage fail stops at the
 * first (exit status 3); continue keeps concealed frames, drops what the
 * decoder rejects, and gives the total at the end. A preview session always
 * goes on and prints nothing. */

#define DMG_TAG ((void *)&g_dmg_tag)
static const char g_dmg_tag = 0;
#define SPOT_GAP 1.0    /* s */

static double us_to_t(const enc_ctx *c, int64_t us)
{
    const double t = (double)(us - c->vstart_us) / AV_TIME_BASE;
    return t > 0 ? t : 0;
}

/* a packet's time (s since the video start), NAN if it has none */
static double pkt_time(const enc_ctx *c, const AVPacket *pkt)
{
    const int64_t ts = pkt->pts != AV_NOPTS_VALUE ? pkt->pts : pkt->dts;
    if (ts == AV_NOPTS_VALUE) return NAN;
    return us_to_t(c, av_rescale_q(ts, c->ifmt->streams[pkt->stream_index]
                                       ->time_base, AV_TIME_BASE_Q));
}

/* [h:]mm:ss.s */
static const char *clock_str(double t, char *buf, size_t n)
{
    const long long ds = (long long)(t * 10 + 0.5);
    const long long s = ds / 10;
    if (s >= 3600)
        snprintf(buf, n, "%lld:%02lld:%02lld.%lld", s / 3600, s / 60 % 60,
                 s % 60, ds % 10);
    else
        snprintf(buf, n, "%lld:%02lld.%lld", s / 60, s % 60, ds % 10);
    return buf;
}

static void damage_report(enc_ctx *c, double t, int64_t frame)
{
    if (c->spot_open && t >= c->spot_first - SPOT_GAP &&
        t <= c->spot_last + SPOT_GAP) {
        if (t > c->spot_last) c->spot_last = t;
        return;                         /* the same spot */
    }
    c->spot_open = 1;
    c->spot_first = c->spot_last = t;
    c->damage_spots++;
    if (g_progress == PROG_JSON)
        printf("{\"damage\":{\"t\":%.3f,\"frame\":%lld}}\n", t, (long long)frame);
    else
        printf("DAMAGE t=%.3f frame=%lld\n", t, (long long)frame);
    fflush(stdout);
    char clk[32], what[64];
    clock_str(t, clk, sizeof clk);
    if (frame >= 0)
        snprintf(what, sizeof what, "frame %lld", (long long)frame);
    else
        snprintf(what, sizeof what, "damaged data, no frame");
    if (c->o.on_damage == DAMAGE_FAIL) {
        loge("damaged video at %s (%s); stopping (--on-damage continue "
             "encodes past damage)", clk, what);
        c->damage_stop = 1;
    } else if (c->damage_spots <= 10) {
        loge("damaged video at %s (%s); continuing", clk, what);
        if (c->damage_spots == 10)
            loge("further damaged spots are listed on stdout only");
    }
}

/* damage found in a packet: wait a few frames for the frame it hit */
static void damage_suspect(enc_ctx *c, double t)
{
    if (c->o.session) return;
    if (c->dec_frames == 0) return;     /* not damage, see above */
    if (isnan(t)) t = c->pkt_t;
    if (c->pend) return;
    c->pend = 1;
    c->pend_t = t;
    c->pend_wait = FFMIN(FFMAX(c->dec->thread_count, 1) +
                         c->dec->has_b_frames + 1, 64);
}

static void damage_pend_flush(enc_ctx *c)
{
    if (c->pend) {
        c->pend = 0;
        damage_report(c, c->pend_t, -1);
    }
}

/* what the decoder and the demuxer logged since the last look */
static void damage_logs(enc_ctx *c, double t)
{
    const long d = ERR_GET(g_dec_errs), m = ERR_GET(g_demux_errs);
    if (d != c->dec_errs_seen || m != c->demux_errs_seen)
        damage_suspect(c, t);
    c->dec_errs_seen = d;
    c->demux_errs_seen = m;
}

/* an avcodec_send_packet / avcodec_receive_frame error: 0 if it is damaged
 * data (dropped), -1 (logged) if it is fatal */
static int dec_error(enc_ctx *c, int r, const char *what)
{
    if (r == AVERROR_INVALIDDATA) {
        damage_suspect(c, c->pkt_t);
        return 0;
    }
    char b[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(r, b, sizeof b);
    loge("video %s failed: %s", what, b);
    return -1;
}

/* a frame out of the decoder (the next source frame) */
static void damage_frame(enc_ctx *c, const AVFrame *fr)
{
    const int64_t idx = c->dec_frames++;
    const int64_t ts = fr->best_effort_timestamp;
    c->frame_t = ts != AV_NOPTS_VALUE
               ? us_to_t(c, av_rescale_q(ts, c->dec->pkt_timebase,
                                         AV_TIME_BASE_Q))
               : c->frame_t + (c->src_fps.num > 0 ? av_q2d(av_inv_q(c->src_fps))
                                                  : 0);
    if (c->o.session) return;
    if (fr->decode_error_flags || (fr->flags & AV_FRAME_FLAG_CORRUPT) ||
        fr->opaque == DMG_TAG) {
        c->pend = 0;
        damage_report(c, c->frame_t, idx);
    } else if (c->pend && --c->pend_wait <= 0) {
        damage_pend_flush(c);
    }
}

/* av_read_frame returned r < 0: 0 = the end of the input, 1 = damaged
 * container data, -1 = a read error (logged) */
static int read_result(enc_ctx *c, int r)
{
    if (c->ifmt->pb && c->ifmt->pb->error < 0)
        r = c->ifmt->pb->error;         /* demuxers often say EOF for it */
    if (r == AVERROR_EOF) return 0;
    if (r == AVERROR_INVALIDDATA) return 1;
    char b[AV_ERROR_MAX_STRING_SIZE], clk[32];
    av_strerror(r, b, sizeof b);
    loge("reading the input failed after %s: %s",
         clock_str(c->pkt_t, clk, sizeof clk), b);
    return -1;
}

/* ---- --deinterlace: libavfilter graph on the decoded frames -------------- */

/* Build the deinterlace graph for decoded frames of format fmt. IVTC is the
 * usual film-on-video recipe: repeatfields turns soft telecine (MPEG-2
 * repeat-field flags) into hard telecine, fieldmatch rebuilds the film
 * frames, bwdif cleans the frames left combed, and on 30 fps sources
 * decimate drops the duplicate of each 3:2 cycle (PAL sources have no
 * pulldown to remove). Deinterlace is bwdif at field rate: one frame per
 * field, so 29.97i becomes 59.94p. */
static int build_deint(enc_ctx *c, int fmt, int spc, int range)
{
    avfilter_graph_free(&c->dgraph);
    c->dgraph = avfilter_graph_alloc();
    if (!c->dgraph) return -1;
    const AVCodecParameters *par = c->ifmt->streams[c->vstream]->codecpar;
    char args[256];
    snprintf(args, sizeof args,
             "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:frame_rate=%d/%d:"
             "pixel_aspect=%d/%d:colorspace=%d:range=%d",
             par->width, par->height, fmt,
             c->dec_tb.num, c->dec_tb.den, c->dec_fps.num, c->dec_fps.den,
             c->src_sar.num > 0 ? c->src_sar.num : 1,
             c->src_sar.num > 0 ? c->src_sar.den : 1, spc, range);
    AV(avfilter_graph_create_filter(&c->dsrc, avfilter_get_by_name("buffer"),
                                    "in", args, NULL, c->dgraph));
    AV(avfilter_graph_create_filter(&c->dsink,
                                    avfilter_get_by_name("buffersink"), "out",
                                    NULL, NULL, c->dgraph));
    const double fps = av_q2d(c->dec_fps);
    const char *chain;
    if (c->o.deint == DEINT_BWDIF)
        chain = "bwdif=mode=send_field:parity=auto:deint=all";
    else if (fps > 29 && fps < 31)
        chain = "repeatfields,fieldmatch=order=auto:combmatch=full,"
                "bwdif=mode=send_frame:parity=auto:deint=interlaced,decimate";
    else
        chain = "repeatfields,fieldmatch=order=auto:combmatch=full,"
                "bwdif=mode=send_frame:parity=auto:deint=interlaced";
    AVFilterInOut *outs = avfilter_inout_alloc(), *ins = avfilter_inout_alloc();
    if (!outs || !ins) {
        avfilter_inout_free(&outs); avfilter_inout_free(&ins);
        return -1;
    }
    outs->name = av_strdup("in");  outs->filter_ctx = c->dsrc;  outs->pad_idx = 0;
    ins->name  = av_strdup("out"); ins->filter_ctx  = c->dsink; ins->pad_idx  = 0;
    int r = avfilter_graph_parse_ptr(c->dgraph, chain, &ins, &outs, NULL);
    avfilter_inout_free(&outs);
    avfilter_inout_free(&ins);
    AV(r);
    AV(avfilter_graph_config(c->dgraph, NULL));
    c->dg_fmt = fmt;
    c->dg_spc = spc;
    c->dg_range = range;
    return 0;
fail:
    loge("deinterlace filter setup failed");
    return -1;
}

/* Set up --deinterlace before libaji is configured: the graph's output
 * rate (field rate, or film rate after IVTC) is the rate everything
 * downstream sees, chain selection included. */
static int setup_deint(enc_ctx *c)
{
    AVStream *vst = c->ifmt->streams[c->vstream];
    c->dec_tb  = vst->time_base;
    c->dec_fps = c->out_fps;
    if (build_deint(c, vst->codecpar->format, vst->codecpar->color_space,
                    vst->codecpar->color_range) < 0)
        return -1;
    const AVRational tb  = av_buffersink_get_time_base(c->dsink);
    const AVRational fps = av_buffersink_get_frame_rate(c->dsink);
    /* frames arrive in the graph's time base; the output keeps the finer of
     * it and the stream's: decimate's is one film frame, which would round
     * the source's start offset (audio sync) away */
    c->in_tb = tb;
    if (av_cmp_q(tb, c->src_tb) < 0)
        c->src_tb = tb;
    if (fps.num > 0 && fps.den > 0) {
        c->src_frames_est = av_rescale(c->src_frames_est, (int64_t)fps.num *
                                       c->dec_fps.den,
                                       (int64_t)fps.den * c->dec_fps.num);
        c->src_fps = c->out_fps = fps;
    }
    loge("%s: %.3f fps -> %.3f fps",
         c->o.deint == DEINT_IVTC ? "inverse telecine" : "deinterlace (bwdif)",
         av_q2d(c->dec_fps), av_q2d(c->out_fps));
    return 0;
}

static int push_frame(enc_ctx *c, AVFrame *fr);

/* one decoded frame through the deinterlace graph (NULL flushes it); the
 * frames it puts out go on into the pipeline */
static void hdr_capture(enc_ctx *c, const AVFrame *fr);

static int push_decoded(enc_ctx *c, AVFrame *fr)
{
    if (fr && !c->hdr_seen)
        hdr_capture(c, fr);
    if (!c->dgraph)
        return fr ? push_frame(c, fr) : 0;
    if (fr) {
        /* the container's description can differ from the decoded frames:
         * rebuild for the first frame's (the output rate stays the same);
         * later the graph's is kept, as the filters can't change mid-stream */
        if (!c->dg_fed && (fr->format != c->dg_fmt ||
                           fr->colorspace != c->dg_spc ||
                           fr->color_range != c->dg_range) &&
            build_deint(c, fr->format, fr->colorspace, fr->color_range) < 0) {
            av_frame_free(&fr);
            return -1;
        }
        c->dg_fed = 1;
        fr->colorspace = c->dg_spc;
        fr->color_range = c->dg_range;
        fr->pts = fr->best_effort_timestamp;
    }
    int r = av_buffersrc_add_frame(c->dsrc, fr);
    av_frame_free(&fr);
    if (r < 0) { AV(r); }
    for (;;) {
        AVFrame *out = av_frame_alloc();
        if (!out) return -1;
        r = av_buffersink_get_frame(c->dsink, out);
        if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) {
            av_frame_free(&out);
            return 0;
        }
        if (r < 0) { av_frame_free(&out); AV(r); }
        out->best_effort_timestamp = out->pts;   /* in the graph's time base */
        if (push_frame(c, out) < 0) return -1;
    }
fail:
    return -1;
}

/* ---- HDR metadata -------------------------------------------------------- */

/* Keep the first decoded frame's HDR10 static metadata. The decoder puts it
 * on every frame, from the bitstream (HEVC/AV1 SEI/OBUs) or the container
 * (codecpar coded side data); upscaling and RIFE don't change it, but
 * libaji's output frames don't carry it. open_encoder hands it to the
 * encoder (decoded_side_data: x265, SVT-AV1, x264, NVENC HEVC/AV1 write it
 * into the bitstream, and libavcodec copies it to the stream's coded side
 * data, which the Matroska and MP4 muxers write), and emit_output puts it
 * on each frame (QSV and AMF read it there). Dolby Vision's RPU describes
 * the source encode and can't be carried through a re-encode: it is
 * dropped, with a note. */
static void hdr_capture(enc_ctx *c, const AVFrame *fr)
{
    static const enum AVFrameSideDataType types[] = {
        AV_FRAME_DATA_MASTERING_DISPLAY_METADATA,
        AV_FRAME_DATA_CONTENT_LIGHT_LEVEL,
    };
    c->hdr_seen = 1;
    for (size_t i = 0; i < FF_ARRAY_ELEMS(types); i++) {
        const AVFrameSideData *sd = av_frame_get_side_data(fr, types[i]);
        if (sd && av_frame_side_data_clone(&c->hdr_sd, &c->nb_hdr_sd, sd,
                                           AV_FRAME_SIDE_DATA_FLAG_UNIQUE) < 0)
            loge("HDR metadata: out of memory; not passed on");
    }
    for (int i = 0; i < c->nb_hdr_sd; i++) {
        const AVFrameSideData *sd = c->hdr_sd[i];
        if (sd->type == AV_FRAME_DATA_MASTERING_DISPLAY_METADATA) {
            const AVMasteringDisplayMetadata *m = (const void *)sd->data;
            loge("HDR mastering display: %s%s, luminance %.4f-%.0f cd/m2",
                 m->has_primaries ? "primaries set" : "no primaries",
                 m->has_luminance ? "" : ", no luminance",
                 m->has_luminance ? av_q2d(m->min_luminance) : 0,
                 m->has_luminance ? av_q2d(m->max_luminance) : 0);
        } else {
            const AVContentLightMetadata *l = (const void *)sd->data;
            loge("HDR content light level: MaxCLL %u, MaxFALL %u cd/m2",
                 l->MaxCLL, l->MaxFALL);
        }
    }

    const AVCodecParameters *par = c->ifmt->streams[c->vstream]->codecpar;
    const AVPacketSideData *dv = av_packet_side_data_get(
        par->coded_side_data, par->nb_coded_side_data, AV_PKT_DATA_DOVI_CONF);
    if (dv) {
        const AVDOVIDecoderConfigurationRecord *cfg = (const void *)dv->data;
        const int compat = cfg->dv_bl_signal_compatibility_id;
        loge("Dolby Vision profile %d: its metadata (RPU) can't be carried "
             "through a re-encode and is dropped; %s", cfg->dv_profile,
             compat == 1 || compat == 6 ? "the output is the HDR10 base layer"
             : compat == 4 ? "the output is the HLG base layer"
             : compat == 2 ? "the output is the SDR base layer"
             : "this base layer has no HDR10/HLG/SDR fallback, so its "
               "colours will be wrong");
    } else if (av_frame_get_side_data(fr, AV_FRAME_DATA_DOVI_RPU_BUFFER) ||
               av_frame_get_side_data(fr, AV_FRAME_DATA_DOVI_METADATA)) {
        loge("Dolby Vision: its metadata (RPU) can't be carried through a "
             "re-encode and is dropped; the output is the base layer");
    }
}

/* ---- libaji init --------------------------------------------------------- */

/* Reduce a RIFE factor num/den in place and check it can be scheduled: at
 * least 1, and small enough that the output time base (src_tb / num) stays
 * exact. Logs why and returns -1 if not. */
static int rife_factor_ok(const enc_ctx *c, int *num, int *den)
{
    const int n0 = *num, d0 = *den;
    if (n0 <= 0 || d0 <= 0 || n0 < d0) {
        loge("RIFE factor %d/%d is not supported: it must be 1 or more "
             "(e.g. 2, 3/2 or 5/2)", n0, d0);
        return -1;
    }
    int a = n0, b = d0;
    while (b) { int t = a % b; a = b; b = t; }
    *num = n0 / a;
    *den = d0 / a;
    if ((int64_t)c->src_tb.den * *num > INT_MAX) {
        loge("RIFE factor %d/%d is too large for the source time base %d/%d",
             n0, d0, c->src_tb.num, c->src_tb.den);
        return -1;
    }
    return 0;
}

static int init_aji(enc_ctx *c)
{
    c->out_tb = c->src_tb;     /* RIFE refines it below */
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
    /* the factor the conf's chain asks for, checked before aji_configure
     * builds engines (minutes, the first time) */
    int qnum = 0, qden = 0;
    if (aji_rife_factor_for_config(&p, c->src_w, c->src_h, av_q2d(c->out_fps),
                                   &qnum, &qden) == 1 &&
        rife_factor_ok(c, &qnum, &qden) < 0)
        return -1;
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
    /* the source's true geometry at the upscale ratio (crops the padding of
     * an odd source; encoders need even dims) */
    c->crop_w = c->up_w;
    c->crop_h = c->up_h;
    if (c->pad_r)
        c->crop_w = (int)((int64_t)c->up_w * (c->src_w - 1) / c->src_w) & ~1;
    if (c->pad_b)
        c->crop_h = (int)((int64_t)c->up_h * (c->src_h - 1) / c->src_h) & ~1;
    /* final encode dims default to those; --final-resize adjusts */
    c->out_w = c->crop_w;
    c->out_h = c->crop_h;
    if (c->sq_after) {
        /* --square-pixels after: the final resize stretches the narrow axis */
        if (c->src_sar.num > c->src_sar.den)
            c->out_w = ((int)av_rescale(c->out_w, c->src_sar.num,
                                        c->src_sar.den) + 1) & ~1;
        else
            c->out_h = ((int)av_rescale(c->out_h, c->src_sar.den,
                                        c->src_sar.num) + 1) & ~1;
    }

    if (aji_rife_factor(c->aji, &c->rnum, &c->rden)) {
        /* the engine's word is final (the check above read the conf) */
        if (rife_factor_ok(c, &c->rnum, &c->rden) < 0) return -1;
        c->rife = 1;
        loge("RIFE: %d/%d interpolation", c->rnum, c->rden);
        c->out_fps = av_mul_q(c->out_fps, (AVRational){c->rnum, c->rden});
        c->out_tb = av_div_q(c->src_tb, (AVRational){c->rnum, 1});
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

/* Fill the padding of an odd-size source (f is the padded frame, w x h the
 * source size) by replicating the last column and row, per plane. */
static void pad_edges(AVFrame *f, int w, int h)
{
    const AVPixFmtDescriptor *d = av_pix_fmt_desc_get(f->format);
    for (int p = 0; p < 4 && f->data[p]; p++) {
        int step = 0, chroma = 0;
        for (int i = 0; i < d->nb_components; i++) {
            if (d->comp[i].plane == p) {
                step = d->comp[i].step;
                chroma = i == 1 || i == 2;
                break;
            }
        }
        if (!step)
            continue;
        const int sw = chroma ? AV_CEIL_RSHIFT(w, d->log2_chroma_w) : w;
        const int sh = chroma ? AV_CEIL_RSHIFT(h, d->log2_chroma_h) : h;
        const int dw = chroma ? AV_CEIL_RSHIFT(f->width, d->log2_chroma_w)
                              : f->width;
        const int dh = chroma ? AV_CEIL_RSHIFT(f->height, d->log2_chroma_h)
                              : f->height;
        uint8_t *base = f->data[p];
        const int ls = f->linesize[p];
        for (int y = 0; y < sh; y++)
            for (int x = sw; x < dw; x++)
                memcpy(base + (size_t)y * ls + (size_t)x * step,
                       base + (size_t)y * ls + (size_t)(sw - 1) * step, step);
        for (int y = sh; y < dh; y++)
            memcpy(base + (size_t)y * ls, base + (size_t)(sh - 1) * ls,
                   (size_t)dw * step);
    }
}

/* Hand libaji a device frame in its input format. NVDEC frames already are
 * one (zero-copy); CPU-decoded frames (always, on DirectML), and NVDEC
 * frames in a layout libaji does not read, are repacked on the host and
 * uploaded - stretched to square pixels there too with --square-pixels
 * before (DVD-sized frames, so the host is cheap). Takes ownership. */
static AVFrame *to_device(enc_ctx *c, AVFrame *fr)
{
    const enum AVPixelFormat sw = src_sw_fmt(c);
    const int pad = c->pad_r || c->pad_b;
    if (!pad && !c->sq_before && fr->format == AV_PIX_FMT_CUDA &&
        fr->hw_frames_ctx &&
        ((AVHWFramesContext *)fr->hw_frames_ctx->data)->sw_format == sw)
        return fr;

    AVFrame *host = NULL, *packed = NULL, *padded = NULL, *dev = NULL;
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
    if (pad) {
        /* pad in the source's own format first (replicating the last column
         * and row), so the conversion below runs even -> even: converting
         * first would resample an odd-width chroma plane at a ratio just off
         * 2:1 and shift the colors of the whole frame */
        padded = av_frame_alloc();
        if (!padded) goto fail;
        padded->format = src->format;
        padded->width  = c->src_w;
        padded->height = c->src_h;
        if (av_frame_get_buffer(padded, 0) < 0) goto fail;
        av_image_copy(padded->data, padded->linesize,
                      (const uint8_t **)src->data, src->linesize,
                      src->format, src->width, src->height);
        pad_edges(padded, src->width, src->height);
        src = padded;
    }
    if (src->format != sw || c->sq_before) {
        const int dw = c->sq_before ? c->src_w : src->width;
        const int dh = c->sq_before ? c->src_h : src->height;
        c->in_sws = sws_getCachedContext(c->in_sws, src->width, src->height,
                                         src->format, dw, dh, sw,
                                         c->sq_before ? SWS_SPLINE : SWS_BICUBIC,
                                         NULL, NULL, NULL);
        packed = av_frame_alloc();
        if (!c->in_sws || !packed) goto fail;
        if (c->src_rgb) {
            /* full-range RGB -> BT.709 limited YUV, as tagged in open_input */
            const int *bt709 = sws_getCoefficients(SWS_CS_ITU709);
            sws_setColorspaceDetails(c->in_sws, bt709, 1, bt709, 0,
                                     0, 1 << 16, 1 << 16);
        }
        packed->format = sw;
        packed->width  = dw;
        packed->height = dh;
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
    av_frame_free(&padded);
    av_frame_free(&host);
    av_frame_free(&fr);
    return dev;
fail:
    av_frame_free(&dev);
    av_frame_free(&packed);
    av_frame_free(&padded);
    av_frame_free(&host);
    av_frame_free(&fr);
    return NULL;
}

/* ---- output: encoder, muxer, streams ------------------------------------- */

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
    } else if (!strcmp(vcodec, "hevc_nvenc")) {
        p = pf == OUT_420P10 ? "main10" : NULL;   /* nvenc is 4:2:0 only here */
    } else if (!strcmp(vcodec, "h264_nvenc")) {
        p = pf == OUT_420P10 ? "high10" : NULL;
    } else if (!strcmp(vcodec, "hevc_amf") || !strcmp(vcodec, "hevc_qsv")) {
        p = pf == OUT_420P10 ? "main10" : NULL;
    }   /* av1_*: the "main" profile covers 10-bit; h264_amf/h264_qsv have no
         * 10-bit (rejected in main) */
    if (p)
        av_dict_set(opts, "profile", p, 0);
}

/* AMF opens on the default adapter (DXGI adapter 0) unless handed a device,
 * so with an NVIDIA card as the default and an AMD iGPU beside it, it would
 * fail. Point it at the upscaling GPU when that is AMD (`prefer`, a DXGI index,
 * -1 for none), else the first AMD adapter (libav derives its AMF device from
 * the D3D11VA one): on an AMD laptop the first is the iGPU, while DirectML
 * upscales on the dGPU. No AMD adapter: leave it alone and let AMF report that. */
static void amf_pick_adapter(AVCodecContext *enc, int prefer)
{
#ifdef _WIN32
    gpu_adapter g[16];
    int n = list_gpus(g, 16), idx = -1;
    for (int i = 0; i < n && idx < 0; i++)
        if (g[i].vendor == 0x1002) idx = g[i].index;
    if (prefer >= 0)
        idx = prefer;
    if (idx <= 0)       /* none, or already the default */
        return;
    char s[16];
    snprintf(s, sizeof s, "%d", idx);
    AVBufferRef *dev = NULL;
    if (av_hwdevice_ctx_create(&dev, AV_HWDEVICE_TYPE_D3D11VA, s, NULL, 0) < 0)
        return;
    enc->hw_device_ctx = dev;       /* the codec context owns the ref */
#else
    (void)enc; (void)prefer;
#endif
}

/* QSV: the oneVPL dispatcher finds an Intel GPU itself, the first one, which
 * with an Intel iGPU and an Arc card is usually the iGPU. When DirectML
 * upscales on an Intel GPU (`prefer`, a DXGI index), open QSV on that one,
 * through a D3D11VA child device on its adapter; otherwise leave it alone. */
static void qsv_pick_adapter(AVCodecContext *enc, int prefer)
{
#ifdef _WIN32
    if (prefer < 0)
        return;
    char s[16];
    snprintf(s, sizeof s, "%d", prefer);
    AVDictionary *opts = NULL;
    av_dict_set(&opts, "child_device_type", "d3d11va", 0);
    av_dict_set(&opts, "child_device", s, 0);
    AVBufferRef *dev = NULL;
    int r = av_hwdevice_ctx_create(&dev, AV_HWDEVICE_TYPE_QSV, "hw", opts, 0);
    av_dict_free(&opts);
    if (r < 0) {
        loge("QSV device on adapter %d failed; letting QSV choose", prefer);
        return;
    }
    enc->hw_device_ctx = dev;       /* the codec context owns the ref */
#else
    (void)enc; (void)prefer;
#endif
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
    c->enc->time_base = c->out_tb;
    c->enc->framerate = c->out_fps;
    c->enc->colorspace            = c->color_space;
    c->enc->color_range           = c->color_range;
    c->enc->color_primaries       = c->color_pri;
    c->enc->color_trc             = c->color_trc;
    c->enc->chroma_sample_location = c->chroma_loc;
    if (c->out_sar.num > 0 && c->out_sar.den > 0)
        c->enc->sample_aspect_ratio = c->out_sar;

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
    /* the encoder runs on the upscaling GPU when it's the same vendor */
    if (is_amf(c->o.vcodec))
        amf_pick_adapter(c->enc, c->gpu_vendor == 0x1002 ? c->gpu_index : -1);
    if (is_qsv(c->o.vcodec))
        qsv_pick_adapter(c->enc, c->gpu_vendor == 0x8086 ? c->gpu_index : -1);
    if (c->ofmt && (c->ofmt->oformat->flags & AVFMT_GLOBALHEADER))
        c->enc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    AVDictionary *opts = NULL;
    parse_vquality(c->o.vquality, &opts);
    inject_profile(c->o.vcodec, c->o.pix_fmt, &opts);
    /* libavcodec defaults thread_count to 1 (the ffmpeg CLI sets "auto"
     * itself), which leaves ffv1/libx264/libx265 on a single core: ~8x
     * slower for ffv1, ~13x for x264. Hardware encoders ignore it. An
     * explicit -threads in --vquality wins. */
    av_dict_set(&opts, "threads", "auto", AV_DICT_DONT_OVERWRITE);

    for (int i = 0; i < c->nb_hdr_sd; i++)
        if (av_frame_side_data_clone(&c->enc->decoded_side_data,
                                     &c->enc->nb_decoded_side_data,
                                     c->hdr_sd[i],
                                     AV_FRAME_SIDE_DATA_FLAG_UNIQUE) < 0) {
            av_dict_free(&opts);
            loge("HDR metadata: out of memory");
            return -1;
        }

    int r = avcodec_open2(c->enc, codec, &opts);
    AVDictionaryEntry *e = NULL;
    while ((e = av_dict_get(opts, "", e, AV_DICT_IGNORE_SUFFIX)))
        loge("warning: --vquality option '%s=%s' not recognized by %s",
             e->key, e->value, c->o.vcodec);
    av_dict_free(&opts);
    if (r < 0 && is_hw_encoder(c->o.vcodec))
        loge("%s could not be opened: this GPU or its driver may not support "
             "it at this bit depth (aji_encode --probe-encoders lists what "
             "works here)", c->o.vcodec);
    if (r < 0) { AV(r); }
    return 0;
fail:
    return -1;
}

/* --probe-encoders: list the GPUs, then open each GPU encoder at 8- and
 * 10-bit 4:2:0. One machine-readable line each, e.g.
 *   GPU index=0 vendor=nvidia default=1 integrated=0 id=9223372036854808576 name=NVIDIA GeForce RTX 5090
 *   ENCODER name=hevc_qsv yuv420p=1 yuv420p10=1
 * id= is what --device takes. integrated=1 marks integrated graphics (shared
 * system memory), which upscale far slower than a graphics card; the key is
 * left out when that can't be told. On Windows the GPUs are DXGI's, index= is the
 * DXGI adapter index, and default=1 marks the GPU --backend directml runs on
 * without --device (default_gpu; TensorRT takes CUDA device 0). On Linux they
 * are the CUDA devices, index= is the CUDA ordinal, and default=1 marks
 * device 0. name= is last and runs to the end of the line. Opening is what
 * fails on a missing GPU, driver, codec or bit depth (all three vendors
 * create their hardware session in init). Software encoders are always
 * available and not listed. */
static void probe_encoders(void)
{
#ifdef _WIN32
    gpu_adapter g[16];
    int ng = list_gpus(g, 16);
    int def = ng > 0 ? default_gpu(g, ng) : -1;
    for (int i = 0; i < ng; i++) {
        int integrated = gpu_integrated(g[i].luid);
        char flag[16] = "";
        if (integrated >= 0)
            snprintf(flag, sizeof flag, " integrated=%d", integrated);
        printf("GPU index=%d vendor=%s default=%d%s id=%llu name=%s\n", g[i].index,
               vendor_name(g[i].vendor), i == def, flag, (unsigned long long)g[i].id,
               g[i].name);
    }
#else
    int nc = 0;
    if (cuInit(0) == CUDA_SUCCESS && cuDeviceGetCount(&nc) == CUDA_SUCCESS) {
        for (int i = 0; i < nc; i++) {
            CUdevice d;
            char name[128];
            if (cuDeviceGet(&d, i) != CUDA_SUCCESS ||
                cuDeviceGetName(name, sizeof name, d) != CUDA_SUCCESS)
                continue;
            int integrated = 0;
            cuDeviceGetAttribute(&integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, d);
            printf("GPU index=%d vendor=nvidia default=%d integrated=%d id=%llu name=%s\n", i,
                   i == 0, integrated != 0, (unsigned long long)cuda_device_id(d), name);
        }
    }
#endif
    static const char *const names[] = {
        "hevc_nvenc", "h264_nvenc", "av1_nvenc",
        "hevc_amf",   "h264_amf",   "av1_amf",
        "hevc_qsv",   "h264_qsv",   "av1_qsv",
    };
    int level = av_log_get_level();
    av_log_set_level(AV_LOG_QUIET);     /* failures here are the expected case */
    for (size_t n = 0; n < sizeof names / sizeof *names; n++) {
        const AVCodec *codec = avcodec_find_encoder_by_name(names[n]);
        int ok[2] = {0, 0};
        for (int ten = 0; codec && ten < 2; ten++) {
            AVCodecContext *e = avcodec_alloc_context3(codec);
            if (!e) continue;
            e->width     = 1920;
            e->height    = 1080;
            e->time_base = (AVRational){1, 24};
            e->framerate = (AVRational){24, 1};
            e->pix_fmt   = ten ? AV_PIX_FMT_P010LE : AV_PIX_FMT_NV12;
            if (is_amf(names[n]))
                amf_pick_adapter(e, -1);
            AVDictionary *opts = NULL;
            inject_profile(names[n], ten ? OUT_420P10 : OUT_420P8, &opts);
            ok[ten] = avcodec_open2(e, codec, &opts) >= 0;
            av_dict_free(&opts);
            avcodec_free_context(&e);
        }
        printf("ENCODER name=%s yuv420p=%d yuv420p10=%d\n",
               names[n], ok[0], ok[1]);
    }
    fflush(stdout);
    av_log_set_level(level);
}

/* Add the output muxer, the video stream, and stream-copy aux streams. */
/* Whether the output container can carry a stream-copied input stream.
 * Copying everything made whole files fail before the first frame: a DVD
 * VOB's nav packets and a MOV's timecode track (data streams, never copied
 * now: they mean nothing outside their container) and MP4 mov_text
 * subtitles are all rejected by Matroska. Attachments (fonts) only exist in
 * Matroska. Everything else is asked of the muxer; an unknown answer keeps
 * the stream, as before - except audio into Matroska, which is always kept:
 * its query only knows Matroska's native codec IDs and says no to audio it
 * stores fine through the ACM compatibility mode (WMA, ...). */
static int can_copy_stream(enc_ctx *c, const AVStream *in)
{
    const AVCodecParameters *p = in->codecpar;
    const int mkv = !strcmp(c->ofmt->oformat->name, "matroska");
    if (p->codec_type == AVMEDIA_TYPE_ATTACHMENT)
        return mkv;
    if (p->codec_id == AV_CODEC_ID_NONE)
        return 0;
    if (p->codec_type == AVMEDIA_TYPE_AUDIO && mkv)
        return 1;
    return avformat_query_codec(c->ofmt->oformat, p->codec_id,
                                FF_COMPLIANCE_NORMAL) != 0;
}

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

    /* Dispositions (default, forced, ...) aren't carried over by libav the
     * way the ffmpeg CLI does it; without them every Matroska track is
     * written FlagDefault=0 and players stop enabling subtitles. The video
     * keeps only language/title: statistics tags (BPS, NUMBER_OF_FRAMES, ...)
     * describe the source encode, not this one. */
    {
        const AVStream *iv = c->ifmt->streams[c->vstream];
        static const char *const keep[] = { "language", "title" };
        vs->disposition = iv->disposition & ~AV_DISPOSITION_ATTACHED_PIC;
        for (size_t k = 0; k < sizeof(keep) / sizeof(*keep); k++) {
            const AVDictionaryEntry *e = av_dict_get(iv->metadata, keep[k],
                                                     NULL, 0);
            if (e) av_dict_set(&vs->metadata, keep[k], e->value, 0);
        }
    }

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
            t != AVMEDIA_TYPE_ATTACHMENT)
            continue;
        if (!can_copy_stream(c, in)) {
            loge("dropping stream #%u (%s %s): the output container can't "
                 "hold it", i, av_get_media_type_string(t),
                 avcodec_get_name(in->codecpar->codec_id));
            continue;
        }
        AVStream *out = avformat_new_stream(c->ofmt, NULL);
        if (!out) { loge("new aux stream failed"); return -1; }
        AV(avcodec_parameters_copy(out->codecpar, in->codecpar));
        out->codecpar->codec_tag = 0;
        out->time_base = in->time_base;
        av_dict_copy(&out->metadata, in->metadata, 0);
        out->disposition = in->disposition;
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

/* Write one stream-copied packet (in its input stream's time base), shifted
 * by the input start like the video, then rescaled to its output stream.
 * Takes ownership. */
static void write_aux(enc_ctx *c, AVPacket *pkt, int in_idx)
{
    const AVStream *in = c->ifmt->streams[in_idx];
    const int oidx = c->smap[in_idx];
    const int64_t off = av_rescale_q(c->start_us, AV_TIME_BASE_Q,
                                     in->time_base);
    if (pkt->pts != AV_NOPTS_VALUE) pkt->pts -= off;
    if (pkt->dts != AV_NOPTS_VALUE) pkt->dts -= off;
    pkt->stream_index = oidx;
    av_packet_rescale_ts(pkt, in->time_base, c->ofmt->streams[oidx]->time_base);
    av_interleaved_write_frame(c->ofmt, pkt);
    av_packet_free(&pkt);
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
    vs->sample_aspect_ratio = c->enc->sample_aspect_ratio;

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
        write_aux(c, c->aux_buf[i], c->aux_src[i]);
        c->aux_buf[i] = NULL;
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
        /* the hw frames context has to be set before the buffer source is
         * initialized: libavfilter 12 rejects a CUDA pix_fmt without one */
        c->buf_src = avfilter_graph_alloc_filter(c->graph,
                avfilter_get_by_name("buffer"), "in");
        if (!c->buf_src) { loge("buffer source alloc failed"); goto fail; }
        AVBufferSrcParameters *bp = av_buffersrc_parameters_alloc();
        if (!bp) goto fail;
        bp->hw_frames_ctx = first->hw_frames_ctx;
        int pr = av_buffersrc_parameters_set(c->buf_src, bp);
        av_free(bp);
        AV(pr);
        AV(avfilter_init_str(c->buf_src, args));

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
    return c->o.final_h > 0 || (c->o.final_pct > 0 && c->o.final_pct != 100) ||
           c->sq_after;
}

/* ---- --preview / --preview-session: still images for a GUI --------------- */

static int ends_with_ci(const char *s, const char *suffix)
{
    const size_t n = strlen(s), m = strlen(suffix);
    return n >= m && !av_strcasecmp(s + n - m, suffix);
}

/* Encode f (device or host, any format) as an image: PNG (full quality, for
 * judging an upscale) or JPEG (4:4:4 when full, else 4:2:0). width > 0
 * scales it to that width, height following the display aspect (sar is
 * the frame's pixel aspect); 0 keeps its size, or takes full_w x full_h when
 * those are set. */
static AVPacket *encode_image(enc_ctx *c, img_enc *ie, const AVFrame *f,
                              AVRational sar, int png, int width, int full_w,
                              int full_h)
{
    AVFrame *dl = NULL, *pv = NULL;
    AVPacket *pkt = NULL;
    int ok = 0;
    if (f->hw_frames_ctx) {
        dl = av_frame_alloc();
        if (!dl || av_hwframe_transfer_data(dl, f, 0) < 0) goto done;
        f = dl;
    }
    /* odd sources: an upscaled frame still carries the padding, crop it off */
    const int sw = (c->pad_r && f->width == c->up_w) ? c->crop_w : f->width;
    const int sh = (c->pad_b && f->height == c->up_h) ? c->crop_h : f->height;
    const enum AVCodecID id = png ? AV_CODEC_ID_PNG : AV_CODEC_ID_MJPEG;
    const int full = width <= 0;
    const enum AVPixelFormat fmt = png ? AV_PIX_FMT_RGB24
                                 : full ? AV_PIX_FMT_YUV444P : AV_PIX_FMT_YUV420P;
    /* anamorphic sources (DVDs) are shown at their display aspect: a full
     * size image is stretched (never squeezed) to it */
    if (sar.num <= 0 || sar.den <= 0)
        sar = (AVRational){1, 1};
    int pw = full_w > 0 ? full_w : sw;
    int ph = full_h > 0 ? full_h : sh;
    if (!full) {
        ph = (int)av_rescale((int64_t)width * ph, sar.den, (int64_t)pw * sar.num);
        pw = width;
    } else if (sar.num > sar.den) {
        pw = (int)av_rescale(pw, sar.num, sar.den);
    } else if (sar.num < sar.den) {
        ph = (int)av_rescale(ph, sar.den, sar.num);
    }
    if (fmt == AV_PIX_FMT_YUV420P) {   /* 4:2:0 JPEG: even dims */
        pw = FFMAX(2, pw & ~1);
        ph = FFMAX(2, ph & ~1);
    }
    pw = FFMAX(1, pw);
    ph = FFMAX(1, ph);
    if (ie->enc && (ie->enc->codec_id != id || ie->enc->pix_fmt != fmt ||
                    ie->enc->width != pw || ie->enc->height != ph))
        avcodec_free_context(&ie->enc);
    if (!ie->enc) {
        const AVCodec *codec = avcodec_find_encoder(id);
        if (!codec || !(ie->enc = avcodec_alloc_context3(codec))) goto done;
        ie->enc->width       = pw;
        ie->enc->height      = ph;
        ie->enc->pix_fmt     = fmt;
        ie->enc->color_range = AVCOL_RANGE_JPEG;
        ie->enc->time_base   = (AVRational){1, 1};
        if (!png) {
            ie->enc->flags |= AV_CODEC_FLAG_QSCALE;
            ie->enc->global_quality = FF_QP2LAMBDA * (full ? 2 : 3);
        }
        if (avcodec_open2(ie->enc, codec, NULL) < 0) {
            avcodec_free_context(&ie->enc);
            loge("preview: %s encoder unavailable", png ? "PNG" : "MJPEG");
            goto done;
        }
    }
    ie->sws = sws_getCachedContext(ie->sws, sw, sh, f->format, pw, ph, fmt,
                                   full ? SWS_BICUBIC : SWS_BILINEAR,
                                   NULL, NULL, NULL);
    if (!ie->sws) goto done;
    /* the stream's matrix and range in; full range out (RGB, or JPEG's
     * BT.601 full-range YUV) */
    const int spc = c->color_space == AVCOL_SPC_BT470BG ||
                    c->color_space == AVCOL_SPC_SMPTE170M ? SWS_CS_ITU601
                  : c->color_space == AVCOL_SPC_BT2020_NCL ||
                    c->color_space == AVCOL_SPC_BT2020_CL ? SWS_CS_BT2020
                  : SWS_CS_ITU709;
    sws_setColorspaceDetails(ie->sws, sws_getCoefficients(spc),
                             c->color_range == AVCOL_RANGE_JPEG,
                             sws_getCoefficients(SWS_CS_ITU601), 1,
                             0, 1 << 16, 1 << 16);
    pv = av_frame_alloc();
    pkt = av_packet_alloc();
    if (!pv || !pkt) goto done;
    pv->format = fmt;
    pv->width = pw;
    pv->height = ph;
    pv->color_range = AVCOL_RANGE_JPEG;
    if (av_frame_get_buffer(pv, 0) < 0) goto done;
    sws_scale(ie->sws, (const uint8_t *const *)f->data, f->linesize,
              0, sh, pv->data, pv->linesize);
    ok = avcodec_send_frame(ie->enc, pv) >= 0 &&
         avcodec_receive_packet(ie->enc, pkt) >= 0;
done:
    av_frame_free(&pv);
    av_frame_free(&dl);
    if (!ok) av_packet_free(&pkt);
    return pkt;
}

/* Write f as an image file (PNG when path ends in .png, else JPEG) of the
 * given width. Written to <path>.tmp, then renamed over <path>, so a reader
 * never sees a partial file. */
static int write_image(enc_ctx *c, const AVFrame *f, const char *path, int width)
{
    AVPacket *pkt = encode_image(c, &c->preview_img, f, c->out_sar,
                                 ends_with_ci(path, ".png"), width, 0, 0);
    FILE *fp = NULL;
    char tmp[4096];
    int ok = 0;
    if (!pkt) goto done;
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    if (!(fp = fopen(tmp, "wb"))) goto done;
    ok = fwrite(pkt->data, 1, pkt->size, fp) == (size_t)pkt->size;
    ok &= fclose(fp) == 0;
    fp = NULL;
#ifdef _WIN32
    /* the UTF-8 manifest makes the A functions take UTF-8, like fopen */
    ok = ok && MoveFileExA(tmp, path, MOVEFILE_REPLACE_EXISTING);
#else
    ok = ok && rename(tmp, path) == 0;
#endif
done:
    if (fp) fclose(fp);
    av_packet_free(&pkt);
    return ok ? 0 : -1;
}

/* --preview during an encode: the latest output frame, downscaled, as a
 * JPEG; a PREVIEW line on stdout says a new one is there. Failures only
 * cost the preview. */
static void write_preview(enc_ctx *c, const AVFrame *f)
{
    const int w = c->o.preview_w > 0 ? c->o.preview_w : 640;
    if (write_image(c, f, c->o.preview, w) == 0 && g_progress != PROG_NONE) {
        printf("PREVIEW frame=%lld\n", (long long)c->out_frames);
        fflush(stdout);
    }
}

/* ---- --preview-session: commands on stdin -------------------------------- *
 *
 * A thread reads command lines; only the latest is kept (a GUI dragging a
 * slider sends many, and only where it stopped matters), numbered so the
 * pipeline can tell a newer one came in and drop what it is doing. */

static struct {
    char     line[256];
    uint64_t n;                /* commands read so far */
    int      eof;
#ifdef _WIN32
    CRITICAL_SECTION   lock;
    CONDITION_VARIABLE cv;
#else
    pthread_mutex_t    lock;
    pthread_cond_t     cv;
#endif
} g_cmd;

#ifdef _WIN32
#define CMD_LOCK()   EnterCriticalSection(&g_cmd.lock)
#define CMD_UNLOCK() LeaveCriticalSection(&g_cmd.lock)
#define CMD_WAKE()   WakeAllConditionVariable(&g_cmd.cv)
#else
#define CMD_LOCK()   pthread_mutex_lock(&g_cmd.lock)
#define CMD_UNLOCK() pthread_mutex_unlock(&g_cmd.lock)
#define CMD_WAKE()   pthread_cond_broadcast(&g_cmd.cv)
#endif

#ifdef _WIN32
static DWORD WINAPI cmd_reader(void *arg)
#else
static void *cmd_reader(void *arg)
#endif
{
    (void)arg;
    /* read(), not stdio: glibc's exit() locks every FILE, stdin's included,
     * and this thread would hold that lock while blocked in fgets(), so the
     * process could never exit */
    char buf[sizeof g_cmd.line], in[512];
    size_t len = 0;
    for (;;) {
#ifdef _WIN32
        const int n = _read(0, in, sizeof in);
#else
        const ssize_t n = read(0, in, sizeof in);
#endif
        if (n <= 0) break;
        for (int i = 0; i < (int)n; i++) {
            if (in[i] != '\n') {
                if (len < sizeof buf - 1) buf[len++] = in[i];
                continue;
            }
            buf[len] = 0;
            buf[strcspn(buf, "\r")] = 0;
            len = 0;
            if (!buf[0]) continue;
            CMD_LOCK();
            memcpy(g_cmd.line, buf, sizeof buf);
            g_cmd.n++;
            CMD_WAKE();
            CMD_UNLOCK();
        }
    }
    CMD_LOCK();
    g_cmd.eof = 1;             /* the GUI went away */
    g_cmd.n++;
    CMD_WAKE();
    CMD_UNLOCK();
    return 0;
}

static int cmd_start(void)
{
#ifdef _WIN32
    InitializeCriticalSection(&g_cmd.lock);
    InitializeConditionVariable(&g_cmd.cv);
    HANDLE t = CreateThread(NULL, 0, cmd_reader, NULL, 0, NULL);
    if (!t) return -1;
    CloseHandle(t);
#else
    pthread_t t;
    pthread_mutex_init(&g_cmd.lock, NULL);
    pthread_cond_init(&g_cmd.cv, NULL);
    if (pthread_create(&t, NULL, cmd_reader, NULL)) return -1;
    pthread_detach(t);
#endif
    return 0;
}

/* Wait until a command newer than `seen` is in, or until wall clock `until`
 * (av_gettime_relative() us; < 0 = no limit). Returns the newest command's
 * number (== seen on timeout) and copies it to line (NULL = don't). */
static uint64_t cmd_wait(uint64_t seen, char *line, int64_t until)
{
    CMD_LOCK();
    while (g_cmd.n == seen) {
        if (until < 0) {
#ifdef _WIN32
            SleepConditionVariableCS(&g_cmd.cv, &g_cmd.lock, INFINITE);
#else
            pthread_cond_wait(&g_cmd.cv, &g_cmd.lock);
#endif
            continue;
        }
        const int64_t left = until - av_gettime_relative();
        if (left <= 0) break;
#ifdef _WIN32
        SleepConditionVariableCS(&g_cmd.cv, &g_cmd.lock,
                                 (DWORD)((left + 999) / 1000));
#else
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        const int64_t ns = ts.tv_nsec + left * 1000;
        ts.tv_sec += ns / 1000000000;
        ts.tv_nsec = ns % 1000000000;
        pthread_cond_timedwait(&g_cmd.cv, &g_cmd.lock, &ts);
#endif
    }
    const uint64_t n = g_cmd.n;
    if (line) {
        if (g_cmd.eof) strcpy(line, "quit");
        else memcpy(line, g_cmd.line, sizeof g_cmd.line);
    }
    CMD_UNLOCK();
    return n;
}

static int cmd_newer(uint64_t seen)
{
    CMD_LOCK();
    const int r = g_cmd.n != seen;
    CMD_UNLOCK();
    return r;
}

/* ---- --preview-session: frames out on stdout ----------------------------- *
 *
 * Text lines, except that a FRAME line is followed by its two images:
 *   READY duration=<ms> width=<w> height=<h> fps=<n>/<d>
 *   FRAME id=<id> pts=<ms> src=<bytes> out=<bytes>  + source JPEG + output JPEG
 *   END id=<id>   (the video ended before the position, or playing reached
 *                  the end)
 * (PROGRESS lines come first while an engine builds.) */

static int sess_send_frame(enc_ctx *c, int64_t pts, const AVPacket *src,
                           const AVPacket *out)
{
    printf("FRAME id=%lld pts=%lld src=%d out=%d\n", c->sess_id,
           (long long)av_rescale_q(pts, c->out_tb, (AVRational){1, 1000}),
           src ? src->size : 0, out->size);
    if ((src && fwrite(src->data, 1, src->size, stdout) != (size_t)src->size) ||
        fwrite(out->data, 1, out->size, stdout) != (size_t)out->size ||
        fflush(stdout)) {
        loge("preview session: stdout closed");
        return -1;
    }
    return 0;
}

static void sess_free_sources(enc_ctx *c)
{
    for (int i = 0; i < c->sess_src_n; i++)
        av_packet_free(&c->sess_src[i].pkt);
    c->sess_src_n = 0;
}

/* a source frame at or after the position: encode it to go out with its
 * upscale */
static int sess_add_source(enc_ctx *c, const AVFrame *fr)
{
    AVPacket *pkt = encode_image(c, &c->sess_src_img, fr, c->src_sar, 0,
                                 c->sess_w, 0, 0);
    if (!pkt) { loge("preview session: source image failed"); return -1; }
    if (c->sess_src_n == FF_ARRAY_ELEMS(c->sess_src)) {
        av_packet_free(&c->sess_src[0].pkt);
        memmove(c->sess_src, c->sess_src + 1,
                (c->sess_src_n - 1) * sizeof *c->sess_src);
        c->sess_src_n--;
    }
    c->sess_src[c->sess_src_n++] = (src_image){fr->pts, pkt};
    return 0;
}

/* emit_output in a session: frames before the target are lead-in; one frame
 * is sent once, playing sends each in real time */
static int sess_emit(enc_ctx *c, AVFrame *f, int64_t pts)
{
    if (c->sess_abort || c->sess_done || c->sess_target == AV_NOPTS_VALUE ||
        pts < c->sess_target)
        return 0;
    if (c->sess_play) {
        const int64_t now = av_gettime_relative();
        int64_t due = c->sess_clock +
                      av_rescale_q(pts - c->sess_pts0, c->out_tb, AV_TIME_BASE_Q);
        if (!c->sess_clock || now - due > 250000) {
            /* the first frame, or running behind: restart the clock here
             * rather than rushing to catch up */
            c->sess_clock = now;
            c->sess_pts0 = pts;
            due = now;
        }
        if (due > now && cmd_wait(c->sess_cmd, NULL, due) != c->sess_cmd) {
            c->sess_abort = 1;
            return 0;
        }
    }
    /* the source frame shown with it: the latest at or before it (RIFE's
     * in-between frames show the one before) */
    while (c->sess_src_n > 1 && c->sess_src[1].pts <= pts) {
        av_packet_free(&c->sess_src[0].pkt);
        memmove(c->sess_src, c->sess_src + 1,
                (c->sess_src_n - 1) * sizeof *c->sess_src);
        c->sess_src_n--;
    }
    AVPacket *out = encode_image(c, &c->preview_img, f, c->out_sar, 0,
                                 c->sess_w, c->out_w, c->out_h);
    if (!out) { loge("preview session: output image failed"); return -1; }
    int r = sess_send_frame(c, pts, c->sess_src_n ? c->sess_src[0].pkt : NULL,
                            out);
    av_packet_free(&out);
    if (!c->sess_play) c->sess_done = 1;
    return r;
}

/* ---- emit one finished output frame: resize -> encode -> mux ------------- */

/* Output frames to expect from src_frames_est source frames. RIFE puts out
 * the points of its output grid (rden/rnum of a source frame apart) from the
 * first source frame through the last, so n frames give
 * floor((n - 1) * rnum / rden) + 1. */
static int64_t out_frames_est(const enc_ctx *c)
{
    const int64_t n = c->src_frames_est;
    if (!c->rife || n <= 0) return n;
    return (n - 1) * c->rnum / c->rden + 1;
}

/* The encode rate for PROGRESS: output frames per second over the last
 * FPS_WINDOW (from the first frame while the encode is younger than that). */
#define FPS_WINDOW 3000000      /* us */

static double encode_fps(enc_ctx *c)
{
    const int64_t now = av_gettime_relative();
    const int n = FF_ARRAY_ELEMS(c->fps_t);
    if (c->fps_count == n) {            /* full: forget the oldest */
        c->fps_head = (c->fps_head + 1) % n;
        c->fps_count--;
    }
    const int in = (c->fps_head + c->fps_count++) % n;
    c->fps_t[in] = now;
    c->fps_n[in] = c->out_frames;
    /* the base: the newest sample at least FPS_WINDOW old, else the oldest */
    while (c->fps_count > 2 &&
           now - c->fps_t[(c->fps_head + 1) % n] >= FPS_WINDOW) {
        c->fps_head = (c->fps_head + 1) % n;
        c->fps_count--;
    }
    const int64_t dt = now - c->fps_t[c->fps_head];
    return dt > 0 ? (double)(c->out_frames - c->fps_n[c->fps_head]) *
                    1e6 / dt : 0;
}

/* Emit one finished output frame at pts (out_tb). Frames arrive in display
 * order (the non-RIFE ring drains FIFO; RIFE emits base then phases in
 * order); the timestamps come from the source frames (frame_pts). */
static int emit_output(enc_ctx *c, AVFrame *cuda_out, int64_t pts)
{
    AVFrame *to_encode = NULL;
    AVFrame *resized = NULL;
    int rc = -1;

    /* a preview session encodes nothing: the frame goes to the GUI (the
     * final resize is done in its image scaler) */
    if (c->o.session) return sess_emit(c, cuda_out, pts);

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
                   (!c->dml || !resize_needed(c)) &&
                   !c->pad_r && !c->pad_b;     /* cropped in the sws path */

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
        /* an odd source's padding is cropped here: read only the top-left
         * crop_w x crop_h (the engine output keeps the padded size) */
        const int in_w = (c->pad_r || c->pad_b) ? c->crop_w : host->width;
        const int in_h = (c->pad_r || c->pad_b) ? c->crop_h : host->height;
        c->sws = sws_getCachedContext(c->sws, in_w, in_h,
                    host->format, sw->width, sw->height, sw->format,
                    SWS_BICUBIC, NULL, NULL, NULL);
        if (!c->sws) { loge("sws_getContext failed"); av_frame_free(&sw);
                       av_frame_free(&dl); goto done; }
        sws_scale(c->sws, (const uint8_t *const *)host->data, host->linesize,
                  0, in_h, sw->data, sw->linesize);
        av_frame_free(&dl);
        to_encode = sw;
        resized = sw;                  /* freed below */
    }

encode:
    if (!c->opened) {
        if (open_encoder(c, to_encode) < 0) goto done;
        if (finalize_output(c) < 0) goto done;
        c->fps_t0 = av_gettime_relative();
        c->fps_count = 1;               /* the rate counts from here */
        c->fps_head = 0;
        c->fps_t[0] = c->fps_t0;
        c->fps_n[0] = 0;
    }
    if (c->o.preview) {
        const int64_t now = av_gettime_relative();
        if (!c->preview_at || now - c->preview_at >= 1000000) {
            c->preview_at = now;
            write_preview(c, to_encode);
        }
    }

    /* Clone before sending so the queued frame owns its own pts (callers
     * reuse frame objects across emits — e.g. RIFE scene-dup) and the
     * underlying buffer stays ref-counted for the encoder's lifetime. */
    {
        AVFrame *sendf = av_frame_clone(to_encode);
        if (!sendf) { loge("av_frame_clone failed"); goto done; }
        sendf->pts = pts;
        for (int i = 0; i < c->nb_hdr_sd; i++)
            if (!av_frame_get_side_data(sendf, c->hdr_sd[i]->type) &&
                av_frame_side_data_clone(&sendf->side_data,
                                          &sendf->nb_side_data, c->hdr_sd[i],
                                          0) < 0) {
                av_frame_free(&sendf);
                loge("HDR metadata: out of memory");
                goto done;
            }
        int sret = avcodec_send_frame(c->enc, sendf);
        av_frame_free(&sendf);
        if (sret < 0) { AV(sret); }
    }
    if (drain_encoder(c, 0) < 0) goto done;

    c->out_frames++;
    if ((c->out_frames & 31) == 0) {
        const int64_t total = out_frames_est(c);
        int pct = total ? (int)(c->out_frames * 100 / total) : 0;
        progress("encode", c->out_frames, total, encode_fps(c),
                 pct > 100 ? 100 : pct);
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
    int r = emit_output(c, s->out, s->out->pts);
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
    /* the output for this frame is inferred later; keep its timestamp */
    c->seq_pts[c->in_seq % FF_ARRAY_ELEMS(c->seq_pts)] = dec->pts;
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
    out->pts = c->seq_pts[c->out_seq % FF_ARRAY_ELEMS(c->seq_pts)];
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
    out->pts = dec->pts;
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
    out->pts = dec->pts;
    return out;
}

/* timestamp of interpolation phase k of n between frames a and b */
static int64_t interp_pts(const AVFrame *a, const AVFrame *b, int k, int n)
{
    return a->pts + (b->pts - a->pts) * k / n;
}

/*
 * The RIFE output schedule (every path; the same grid as mpv's
 * vf_animejanai). Output frames lie on a grid rden/rnum of a source frame
 * apart, starting at the first source frame. Between held frame a and its
 * successor b they fall at a + (b - a) * acc / rnum for acc = rife_acc,
 * rife_acc + rden, ... while acc < rnum; acc == 0 is a itself (the grid
 * lands on a source frame), any other acc an interpolation at
 * t = acc / rnum. Then rife_acc -= rnum carries the remainder into the next
 * pair. An integer factor F (rden == 1) gives acc = 0, 1, ..., F-1 in every
 * pair: a, then F-1 phases at k/F, as before fractions were supported.
 *
 * Timestamps are exact: out_tb = src_tb / rnum, so a source frame d source
 * ticks long is d * rnum output ticks and every grid point of a
 * constant-rate source lands on a whole tick (frame_pts keeps frames at
 * least rnum ticks apart, so in-between timestamps never collide). A
 * variable-rate source places them proportionally within each pair, like
 * mpv.
 *
 * At the end of the stream the last source frame is put out only if the grid
 * lands on it (always for an integer factor); nothing is put out past it, as
 * there is no successor to interpolate towards. So n source frames give
 * floor((n - 1) * rnum / rden) + 1 output frames (out_frames_est).
 */

/* take one upscaled frame (up==NULL flushes): emit the grid points from the
 * held frame up to (not including) this one */
static int rife_step(enc_ctx *c, AVFrame *up)
{
    if (!up) {
        /* the final upscaled frame, if on the grid: no successor to
         * interpolate with */
        if (c->have_prev && c->rife_acc == 0) {
            if (emit_output(c, c->up_prev, c->up_prev->pts) < 0) return -1;
        }
        return 0;
    }

    if (!c->have_prev) {
        c->up_prev = up;
        c->have_prev = 1;
        c->rife_acc = 0;               /* the grid starts on the first frame */
        return 0;
    }
    c->up_cur = up;

    for (; c->rife_acc < c->rnum; c->rife_acc += c->rden) {
        const int acc = c->rife_acc;
        if (acc == 0) {
            /* on source frame N (= up_prev) */
            if (emit_output(c, c->up_prev, c->up_prev->pts) < 0) return -1;
            continue;
        }
        /* between N and N+1 */
        double t = (double)acc / c->rnum;
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
        int er = emit_output(c, emit,
                             interp_pts(c->up_prev, c->up_cur, acc, c->rnum));
        av_frame_free(&interp);
        if (er < 0) return -1;
    }
    c->rife_acc -= c->rnum;

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

/* RIFE-first (dec==NULL flushes): when source frame N+1 arrives, emit the
 * grid points from N up to N+1 (rife_step's schedule): upscaled N if the
 * grid lands on it, and each interpolation between the two, upscaled. A
 * scene change repeats upscaled N, like the default order. */
static int push_rife_first(enc_ctx *c, AVFrame *dec)
{
    if (!dec) {
        if (!c->src_prev || c->rife_acc != 0) return 0;  /* off the grid */
        AVFrame *up = upscale_one(c, c->src_prev);
        int r = up ? emit_output(c, up, up->pts) : -1;
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
        cur->pts = dec->pts;
        av_frame_free(&dec);
        if (r < 0) { av_frame_free(&cur); return -1; }
    }
    if (!c->src_prev) {
        c->src_prev = cur;
        c->rife_acc = 0;               /* the grid starts on the first frame */
        return 0;
    }

    AVFrame *base = NULL;              /* upscaled N, once needed */
    int rc = 0;
    for (; rc == 0 && c->rife_acc < c->rnum; c->rife_acc += c->rden) {
        const int acc = c->rife_acc;
        if (acc == 0) {
            base = upscale_one(c, c->src_prev);
            rc = base ? emit_output(c, base, base->pts) : -1;
            continue;
        }
        AVFrame *interp = alloc_work(c);
        if (!interp) { rc = -1; break; }
        interp->pts = interp_pts(c->src_prev, cur, acc, c->rnum);
        aji_frame a, b, o;
        fill_aji_frame(c, &a, c->src_prev, 0);
        fill_aji_frame(c, &b, cur, 0);
        fill_aji_frame(c, &o, interp, 0);
        int ret = aji_infer_rife(c->aji, &a, &b, (double)acc / c->rnum, &o,
                                 c->stream);
        if (ret == AJI_SCENE) {
            c->scene_dupes++;
            /* off the grid, N itself wasn't put out (or upscaled) */
            if (!base) base = upscale_one(c, c->src_prev);
            rc = base ? emit_output(c, base, interp->pts) : -1;
        } else if (ret != AJI_OK) {
            loge("aji_infer_rife: %d: %s", ret, aji_last_error(c->aji));
            rc = -1;
        } else {
            AVFrame *up = upscale_one(c, interp);
            rc = up ? emit_output(c, up, up->pts) : -1;
            av_frame_free(&up);
        }
        av_frame_free(&interp);
    }
    c->rife_acc -= c->rnum;
    av_frame_free(&base);
    av_frame_free(&c->src_prev);
    c->src_prev = cur;
    return rc;
}

/* A decoded frame's timestamp in out_tb, from the input's start: the
 * decoder's best-effort timestamp (variable frame rate sources keep their
 * timing), filled in where missing and kept strictly increasing, by a whole
 * source frame so RIFE's in-between timestamps stay distinct. */
static int64_t frame_pts(enc_ctx *c, const AVFrame *fr)
{
    if (!c->frame_dur)
        c->frame_dur = FFMAX(1, av_rescale_q(1, av_inv_q(c->src_fps), c->out_tb));
    int64_t pts = fr->best_effort_timestamp;
    if (pts != AV_NOPTS_VALUE)
        pts = av_rescale_q(pts, c->in_tb, c->out_tb) -
              av_rescale_q(c->start_us, AV_TIME_BASE_Q, c->out_tb);
    if (c->last_pts != AV_NOPTS_VALUE) {
        const int64_t min_step = c->rife ? c->rnum : 1;
        if (pts == AV_NOPTS_VALUE)
            pts = c->last_pts + c->frame_dur;
        else if (pts < c->last_pts + min_step)
            pts = c->last_pts + min_step;
    } else if (pts == AV_NOPTS_VALUE) {
        pts = 0;
    }
    c->last_pts = pts;
    return pts;
}

/* one decoded frame into the active pipeline (NULL flushes it) */
static int push_frame(enc_ctx *c, AVFrame *fr)
{
    if (fr)
        fr->pts = frame_pts(c, fr);
    if (fr && c->o.session) {
        /* the first frame at or after the position is the one asked for.
         * Earlier frames are lead-in, and only as many go on through the
         * pipeline as a temporal window or a RIFE pair needs - the rest are
         * dropped rather than upscaled. */
        const int64_t lead = (int64_t)(c->temporal_r + 2) * c->frame_dur;
        if (c->sess_target == AV_NOPTS_VALUE && fr->pts < c->sess_at - lead) {
            av_frame_free(&fr);
            return 0;
        }
        if (c->sess_target == AV_NOPTS_VALUE && fr->pts >= c->sess_at)
            c->sess_target = fr->pts;
        if (c->sess_target != AV_NOPTS_VALUE) {
            if ((c->sess_play || c->sess_after == 0) &&
                sess_add_source(c, fr) < 0) {
                av_frame_free(&fr);
                return -1;
            }
            c->sess_after++;
        }
    }
    if (fr && (!c->passthrough || c->rife_only || c->sq_before) &&
        !(fr = to_device(c, fr)))
        return -1;
    if (!c->rife) return push_upscale(c, fr);
    if (c->rife_only) return rife_step(c, fr);   /* interpolate the frames as-is */
    return c->rife_first ? push_rife_first(c, fr) : push_rife(c, fr);
}

/* ---- main demux/decode loop ---------------------------------------------- */

static int run(enc_ctx *c)
{
    AVPacket *pkt = av_packet_alloc();
    int rc = -1, r, bad_reads = 0;
    if (!pkt) goto done;

    c->dec_errs_seen = ERR_GET(g_dec_errs);
    c->demux_errs_seen = ERR_GET(g_demux_errs);
    g_count_errs = 1;
    for (;;) {
        r = av_read_frame(c->ifmt, pkt);
        if (r == AVERROR(EAGAIN)) { av_usleep(10000); continue; }
        if (r < 0) {
            const int how = read_result(c, r);
            if (how < 0) goto done;
            if (how == 0) break;
            /* damaged container data: the demuxer resyncs on the next
             * read, unless it can't get past it */
            damage_suspect(c, NAN);
            if (c->damage_stop) goto damaged;
            if (++bad_reads == 64) {
                loge("the rest of the input can't be read; finishing");
                break;
            }
            continue;
        }
        bad_reads = 0;
        if (pkt->stream_index == c->vstream) {
            const double t = pkt_time(c, pkt);
            if (!isnan(t)) c->pkt_t = t;
            damage_logs(c, c->pkt_t);   /* the demuxer's, reading it */
            if (pkt->flags & AV_PKT_FLAG_CORRUPT) {
                pkt->opaque = DMG_TAG;  /* its frame is damaged too */
                damage_suspect(c, c->pkt_t);
            }
            r = dec_send(c, pkt);
            if (r < 0 && dec_error(c, r, "decoding") < 0) goto done;
            damage_logs(c, c->pkt_t);
            for (;;) {
                AVFrame *fr = av_frame_alloc();
                if (!fr) goto done;
                r = dec_receive(c, fr);
                if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) {
                    av_frame_free(&fr);
                    break;
                }
                if (r < 0) {
                    av_frame_free(&fr);
                    if (dec_error(c, r, "decoding") < 0) goto done;
                    continue;
                }
                damage_frame(c, fr);
                if (c->damage_stop) { av_frame_free(&fr); goto damaged; }
                /* fr is owned by the pipeline now */
                if (push_decoded(c, fr) < 0) goto done;
            }
            damage_logs(c, c->pkt_t);
            if (c->damage_stop) goto damaged;
        } else {
            if (ERR_GET(g_demux_errs) != c->demux_errs_seen)
                damage_logs(c, pkt_time(c, pkt));
            if (c->damage_stop) goto damaged;
            if (c->smap && c->smap[pkt->stream_index] >= 0) {
                AVPacket *out = av_packet_clone(pkt);
                if (c->opened) {
                    write_aux(c, out, pkt->stream_index);
                } else {
                    /* header not written yet: buffer (in the input time
                     * base; the output stream's is only final after the
                     * header) */
                    if (c->aux_n == c->aux_cap) {
                        c->aux_cap = c->aux_cap ? c->aux_cap * 2 : 16;
                        c->aux_buf = av_realloc_f(c->aux_buf, c->aux_cap,
                                                  sizeof(*c->aux_buf));
                        c->aux_src = av_realloc_f(c->aux_src, c->aux_cap,
                                                  sizeof(*c->aux_src));
                    }
                    c->aux_src[c->aux_n] = pkt->stream_index;
                    c->aux_buf[c->aux_n++] = out;
                }
            }
        }
        av_packet_unref(pkt);
    }

    /* flush decoder */
    r = dec_send(c, NULL);
    if (r < 0 && dec_error(c, r, "decoding") < 0) goto done;
    for (;;) {
        AVFrame *fr = av_frame_alloc();
        if (!fr) goto done;
        r = dec_receive(c, fr);
        if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) { av_frame_free(&fr); break; }
        if (r < 0) {
            av_frame_free(&fr);
            if (dec_error(c, r, "decoding") < 0) goto done;
            continue;
        }
        damage_frame(c, fr);
        if (c->damage_stop) { av_frame_free(&fr); goto damaged; }
        if (push_decoded(c, fr) < 0) goto done;
    }
    damage_logs(c, c->pkt_t);
    damage_pend_flush(c);               /* no frames left to tie it to */
    g_count_errs = 0;
    if (c->damage_stop) goto damaged;
    if (push_decoded(c, NULL) < 0) goto done;   /* flush the deinterlacer */

    /* flush pipeline */
    if (push_frame(c, NULL) < 0) goto done;

    if (!c->opened) { loge("no video frames produced"); goto done; }

    /* flush encoder + trailer */
    if (drain_encoder(c, 1) < 0) goto done;
    AV(av_write_trailer(c->ofmt));

    if (c->rife)
        loge("rife: produced %lld frames (~expected %lld), %lld scene dupes",
             (long long)c->out_frames, (long long)out_frames_est(c),
             (long long)c->scene_dupes);
    if (c->damage_spots)
        loge("damaged video: %lld spot(s), encoded past (concealed frames "
             "kept, data the decoder rejected dropped)",
             (long long)c->damage_spots);
    {   /* the last line: the whole encode's average */
        const int64_t dt = av_gettime_relative() - c->fps_t0;
        progress("encode", c->out_frames, c->out_frames,
                 dt > 0 ? (double)c->out_frames * 1e6 / dt : 0, 100);
    }
    rc = 0;
    goto done;
damaged:
    rc = EXIT_DAMAGED;
done:
fail:
    g_count_errs = 0;
    av_packet_free(&pkt);
    return rc;
}

/* ---- --preview-session: serving requests --------------------------------- */

/* Forget everything in the pipeline (its frames are from before a seek):
 * in-flight upscales are waited on and dropped, the temporal history, RIFE's
 * held frames, the decoder and the deinterlacer start over. */
static int sess_reset(enc_ctx *c)
{
    while (c->ring_count > 0) {
        slot *s = &c->ring[(c->ring_head - c->ring_count + MAX_RING)
                           % MAX_RING];
        if (s->ticket && aji_wait(c->aji, s->ticket) != AJI_OK) {
            loge("aji_wait failed: %s", aji_last_error(c->aji));
            return -1;
        }
        av_frame_free(&s->out);
        av_frame_free(&s->in_ref);
        s->ticket = 0;
        c->ring_count--;
    }
    if (c->aji && sync_aji(c) < 0) return -1;
    av_frame_free(&c->up_prev);
    av_frame_free(&c->up_cur);
    av_frame_free(&c->src_prev);
    c->have_prev = 0;
    if (c->temporal_r) {
        aji_temporal_reset(c->aji);
        c->in_seq = c->out_seq = 0;
    }
    c->last_pts = AV_NOPTS_VALUE;
    dec_flush(c);
    if (c->dgraph && build_deint(c, c->dg_fmt, c->dg_spc, c->dg_range) < 0)
        return -1;
    sess_free_sources(c);
    c->sess_target = AV_NOPTS_VALUE;
    c->sess_after = 0;
    c->sess_done = c->sess_abort = 0;
    c->sess_clock = 0;
    return 0;
}

/* Serve one request from a seek a few seconds before the position (lead-in
 * for temporal windows, RIFE and IVTC's field matching; it lands on the
 * keyframe before). One frame is flushed out as soon as the frames its
 * window needs are in. Returns once it is done, a newer command came in, or
 * at the end of the video. */
static int sess_serve(enc_ctx *c, double sec)
{
    AVPacket *pkt = av_packet_alloc();
    int rc = -1, eof = 0;
    if (!pkt || sess_reset(c) < 0) goto done;
    c->sess_at = av_rescale_q((int64_t)(sec * AV_TIME_BASE), AV_TIME_BASE_Q,
                              c->out_tb);
    const int64_t ts = c->start_us +
                       (int64_t)(FFMAX(sec - 3, 0) * AV_TIME_BASE);
    /* MPEG-PS/TS seek by searching for timestamps, and from the start that
     * lands a GOP late: rewind those to the first byte instead */
    const int rewind = sec <= 3 &&
                       (c->ifmt->iformat->flags & AVFMT_TS_DISCONT) &&
                       !(c->ifmt->iformat->flags & AVFMT_NO_BYTE_SEEK);
    if ((rewind ? avformat_seek_file(c->ifmt, -1, 0, 0, 0, AVSEEK_FLAG_BYTE)
                : avformat_seek_file(c->ifmt, -1, INT64_MIN, ts, ts, 0)) < 0)
        loge("preview session: seek to %.3f s failed", sec);

    while (!c->sess_done && !c->sess_abort) {
        if (!eof) {
            const int r = av_read_frame(c->ifmt, pkt);
            if (r == AVERROR(EAGAIN)) continue;
            if (r < 0) {
                /* a read error ends the session; damaged container data
                 * ends the video here, like its end */
                if (read_result(c, r) < 0) goto done;
                eof = 1;
                dec_send(c, NULL);
            } else if (pkt->stream_index != c->vstream) {
                av_packet_unref(pkt);
                continue;
            } else {
                /* damage never stops a preview (it has no --on-damage):
                 * a rejected packet is dropped, a concealed frame shown */
                const double t = pkt_time(c, pkt);
                if (!isnan(t)) c->pkt_t = t;
                const int s = dec_send(c, pkt);
                av_packet_unref(pkt);
                if (s < 0 && s != AVERROR(EAGAIN) && s != AVERROR_INVALIDDATA)
                    AV(s);
            }
        }
        for (;;) {
            AVFrame *fr = av_frame_alloc();
            const int r = dec_receive(c, fr);
            if (r == AVERROR(EAGAIN)) { av_frame_free(&fr); break; }
            if (r == AVERROR_EOF) {
                av_frame_free(&fr);
                /* the rest of the video: flush the deinterlacer and the
                 * pipeline, then say it ended */
                if (push_decoded(c, NULL) < 0 || push_frame(c, NULL) < 0)
                    goto done;
                if (!c->sess_done && !c->sess_abort) {
                    printf("END id=%lld\n", c->sess_id);
                    fflush(stdout);
                }
                rc = 0;
                goto done;
            }
            if (r == AVERROR_INVALIDDATA) { av_frame_free(&fr); continue; }
            if (r < 0) { av_frame_free(&fr); AV(r); }
            if (push_decoded(c, fr) < 0) goto done;
            if (c->sess_done || c->sess_abort) break;
            /* one frame: its window (the r frames after it) is in, flush */
            if (!c->sess_play && c->sess_target != AV_NOPTS_VALUE &&
                c->sess_after > c->temporal_r) {
                if (push_frame(c, NULL) < 0) goto done;
                break;
            }
        }
        if (cmd_newer(c->sess_cmd)) c->sess_abort = 1;
    }
    rc = 0;
done:
fail:
    av_packet_free(&pkt);
    return rc;
}

/* --preview-session: libaji stays configured (engines loaded) while a GUI's
 * player asks for frames; commands are lines on stdin:
 *   frame <id> <seconds> <width>   the frame at that position (width 0 =
 *                                  full size), with its source frame
 *   play <id> <seconds> <width>    every frame from there on, in real time
 *   stop                           stop playing
 *   quit                           (or end of input) exit
 * A new command replaces the one being served. Replies: see sess_emit. */
static int session(enc_ctx *c)
{
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);   /* the images are binary */
#endif
    if (cmd_start() < 0) { loge("preview session: no stdin thread"); return -1; }
    const int64_t dur = c->ifmt->duration > 0 ? c->ifmt->duration / 1000 : 0;
    printf("READY duration=%lld width=%d height=%d fps=%d/%d\n",
           (long long)dur, c->out_w, c->out_h, c->out_fps.num, c->out_fps.den);
    fflush(stdout);

    char line[sizeof g_cmd.line];
    uint64_t seen = 0;
    for (;;) {
        seen = cmd_wait(seen, line, -1);
        char kind[16] = "";
        long long id = 0;
        double sec = 0;
        int w = 0;
        const int n = sscanf(line, "%15s %lld %lf %d", kind, &id, &sec, &w);
        if (!strcmp(kind, "quit")) return 0;
        if (!strcmp(kind, "stop")) continue;
        const int play = !strcmp(kind, "play");
        if ((!play && strcmp(kind, "frame")) || n < 3) {
            loge("preview session: unknown command '%s'", line);
            continue;
        }
        c->sess_cmd = seen;
        c->sess_id = id;
        c->sess_play = play;
        c->sess_w = FFMAX(w, 0);
        if (sess_serve(c, FFMAX(sec, 0)) < 0) return -1;
    }
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
    avfilter_graph_free(&c->dgraph);
    sws_freeContext(c->preview_img.sws);
    avcodec_free_context(&c->preview_img.enc);
    sws_freeContext(c->sess_src_img.sws);
    avcodec_free_context(&c->sess_src_img.enc);
    sess_free_sources(c);
    if (c->sws) sws_freeContext(c->sws);
    for (int i = 0; i < c->aux_n; i++) av_packet_free(&c->aux_buf[i]);
    av_free(c->aux_buf);
    av_free(c->aux_src);
    av_free(c->smap);
    if (c->aji) aji_destroy(&c->aji);
    if (c->stream) cudaStreamDestroy(c->stream);
    av_buffer_unref(&c->aji_pool);
    av_buffer_unref(&c->src_pool);
    av_buffer_unref(&c->work_pool);
    if (c->enc) avcodec_free_context(&c->enc);
    dec_drop_pending(c);
    dec_drop_held(c);
    if (c->dec) avcodec_free_context(&c->dec);
    av_frame_side_data_free(&c->hdr_sd, &c->nb_hdr_sd);
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
"           [--vcodec hevc_nvenc|h264_nvenc|av1_nvenc|hevc_amf|h264_amf|av1_amf|\n"
"                     hevc_qsv|h264_qsv|av1_qsv|libx265|libx264|ffv1]"
" [--vquality \"<args>\"]\n"
"           [--pix-fmt yuv420p|yuv420p10|yuv444p|yuv444p10]"
" (default yuv420p10; 4:4:4 = software encoder)\n"
"aji_encode --probe-encoders   (which GPU encoders open on this machine)\n"
"           [--final-resize-height H | --final-resize-factor PCT]"
" [--overwrite]\n"
"           [--no-audio] [--no-subs] [--no-chapters]"
" [--progress none|line|json]\n"
"           [--build-only] [--log <f>] [--decoder auto|nvdec|cpu]"
" [--backend tensorrt|tensorrt-rtx|directml]\n"
"           [--device <id>] (a GPU id= from --probe-encoders; default: on\n"
"                           DirectML the high-performance GPU, on TensorRT CUDA device 0)\n"
"           [--deinterlace off|ivtc|bwdif] [--square-pixels off|before|after]\n"
"           [--on-damage fail|continue] (damaged video: fail = stop at the first\n"
"                           damaged spot, exit status 3 (default); continue = encode\n"
"                           past it; each spot prints DAMAGE t=<s> frame=<n|-1>)\n"
"           [--preview <f.jpg> [--preview-width W]]\n"
"           [--preview-session] (no --output: frames for a GUI's player)\n"
"           (direct mode: --engine <e> --max-width W --max-height H)\n");
}

/* Windows 11 power-throttles (EcoQoS) windowless processes whose parent
 * isn't in the foreground, and coarsens their timer resolution: an encode
 * launched by a GUI slows to a crawl once that GUI loses focus or is
 * minimized. Opt this process out of both. */
static void disable_power_throttling(void)
{
#ifdef _WIN32
    PROCESS_POWER_THROTTLING_STATE s = {0};
    s.Version     = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
    s.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED |
                    PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION;
    s.StateMask   = 0;
    SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling,
                          &s, sizeof s);
#endif
}

int main(int argc, char **argv)
{
    disable_power_throttling();
    enc_ctx c = {0};
    c.o.slot = 1;
    c.gpu_index = -1;
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
        else if (!strcmp(a, "--preview-session")) c.o.session = 1;
        else if (!strcmp(a, "--no-zerocopy")) c.o.no_zerocopy = 1;
        else if (!strcmp(a, "--probe-encoders")) { probe_encoders(); return 0; }
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
        else if (!strcmp(a, "--device")) {
            char *end = NULL;
            c.o.device = strtoull(argv[++i], &end, 0);
            if (!end || *end || !c.o.device) {
                loge("--device '%s': expected a GPU id from --probe-encoders", argv[i]);
                return 2;
            }
        }
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
        else if (!strcmp(a, "--deinterlace")) {
            const char *v = argv[++i];
            if (!strcmp(v, "off"))        c.o.deint = DEINT_OFF;
            else if (!strcmp(v, "ivtc"))  c.o.deint = DEINT_IVTC;
            else if (!strcmp(v, "bwdif")) c.o.deint = DEINT_BWDIF;
            else { loge("unknown --deinterlace '%s' (off|ivtc|bwdif)", v); return 2; }
        }
        else if (!strcmp(a, "--square-pixels")) {
            const char *v = argv[++i];
            if (!strcmp(v, "off"))         c.o.square = SQUARE_OFF;
            else if (!strcmp(v, "before")) c.o.square = SQUARE_BEFORE;
            else if (!strcmp(v, "after"))  c.o.square = SQUARE_AFTER;
            else { loge("unknown --square-pixels '%s' (off|before|after)", v);
                   return 2; }
        }
        else if (!strcmp(a, "--on-damage")) {
            const char *v = argv[++i];
            if (!strcmp(v, "fail"))          c.o.on_damage = DAMAGE_FAIL;
            else if (!strcmp(v, "continue")) c.o.on_damage = DAMAGE_CONTINUE;
            else { loge("unknown --on-damage '%s' (fail|continue)", v); return 2; }
        }
        else if (!strcmp(a, "--preview")) c.o.preview = argv[++i];
        else if (!strcmp(a, "--preview-width")) c.o.preview_w = atoi(argv[++i]);
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
    } else if (av_strcasecmp(c.o.backend, "tensorrt") &&
               av_strcasecmp(c.o.backend, "tensorrt-rtx")) {
        /* tensorrt-rtx: CUDA frames, like tensorrt */
        loge("unknown --backend '%s' (tensorrt|tensorrt-rtx|directml)",
             c.o.backend);
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
    if (!c.o.input || (!c.o.build_only && !c.o.session && !c.o.output)) {
        usage(); return 2;
    }
    if (!c.o.conf && !c.o.engine) { loge("need --conf or --engine"); return 2; }
    if (c.o.final_h > 0 && c.o.final_pct > 0) {
        loge("--final-resize-height and --final-resize-factor are exclusive");
        return 2;
    }
    if (c.o.pix_fmt >= OUT_444P8 && is_hw_encoder(c.o.vcodec)) {
        loge("4:4:4 output needs a software encoder (libx265 main444 / "
             "libx264 high444); %s 4:4:4 not supported", c.o.vcodec);
        return 2;
    }
    if (c.o.pix_fmt == OUT_420P10 &&
        (!strcmp(c.o.vcodec, "h264_amf") || !strcmp(c.o.vcodec, "h264_qsv"))) {
        loge("%s has no 10-bit mode; use --pix-fmt yuv420p, or HEVC/AV1 "
             "for 10-bit", c.o.vcodec);
        return 2;
    }
    c.ring_depth = c.o.pipeline_depth;
    if (c.ring_depth < 1) c.ring_depth = 1;
    if (c.ring_depth > MAX_RING - 1) c.ring_depth = MAX_RING - 1;

    int rc = 1;
    if (open_input(&c) < 0) goto out;
    if (c.o.deint && setup_deint(&c) < 0) goto out;
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
    if (c.o.session) {
        rc = session(&c) < 0;
        goto out;
    }
    if (open_output(&c) < 0) goto out;
    rc = run(&c);
    if (rc < 0) rc = 1;

out:
    cleanup(&c);
    if (g_log && g_log != stderr) fclose(g_log);
    return rc;
}
