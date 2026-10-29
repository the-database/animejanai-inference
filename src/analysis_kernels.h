#pragma once
#include "aji.h"
#include "kernels.h"
int aji_analysis_pre(const aji_frame *frame, const int crop[4], int width, int height,
    int inner_w, int inner_h, float *input, unsigned char *color, void *stream);
int aji_analysis_mask(const float *mask, int width, int height, int inner_w, int inner_h,
    uint16_t *output, void *stream);
