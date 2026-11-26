// Shared kernel body. p selects the partial sums and output of one convolution.
    const int pairs = p.coutp_all / 2;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x, npix = (size_t)p.H * p.W;
    if (i >= npix * pairs) return;
    const int ch = (int)(i % pairs) * 2;
    const size_t pix = i / pairs;
    const int y = (int)(pix / p.W), x = (int)(pix % p.W);
    float2 s = make_float2(0.f, 0.f);
    for (int k = 0; k < p.ksplit; k++) {
        const float2 v = *reinterpret_cast<const float2 *>(p.ws + ((size_t)k * npix + pix) * p.coutp_all + ch);
        s.x += v.x;
        s.y += v.y;
    }
    const float2 b = __half22float2(p.bias[ch / 2]);
    if (p.outf) {
        s.x += b.x;
        s.y += b.y;
        if (p.r1f) {
            const float2 r = *reinterpret_cast<const float2 *>(p.r1f + pix * p.r1fs + ch);
            s.x += r.x;
            s.y += r.y;
        }
        if (p.act == 1) {
            s.x = fmaxf(s.x, 0.f);
            s.y = fmaxf(s.y, 0.f);
        } else if (p.act == 2) {
            s.x = s.x < 0.f ? s.x * p.slopef : s.x;
            s.y = s.y < 0.f ? s.y * p.slopef : s.y;
        }
        *reinterpret_cast<float2 *>(p.dstf + pix * p.dstride + ch) = s;
        return;
    }
    half2 v = __floats2half2_rn(s.x + b.x, s.y + b.y);
    v = g_epi(p, v, y, x, ch);
    if (p.R == 0) {
        *reinterpret_cast<half2 *>(p.dst + pix * p.dstride + ch) = v;
    } else {
        const int CHP = p.coutp_all / p.R, sx = ch / CHP, c = ch % CHP;
        *reinterpret_cast<half2 *>(p.dst + ((size_t)(p.R * y + p.sy) * (p.R * p.W) + p.R * x + sx) * p.dstride + c) = v;
    }
