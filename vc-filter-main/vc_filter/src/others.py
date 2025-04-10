# Apr-10-2024
# others.py

import cv2 as cv
import numpy as np


def sobel_accum(height_dft, width_dft):

    # Sobel
    # ---------------------------------------------------------
    array_filter = sobel_kernel(height_dft, width_dft)

    dft_filter = cv.dft(array_filter, flags=cv.DFT_COMPLEX_OUTPUT)

    # DFT{Filter}
    dft_filter_shift = np.fft.fftshift(dft_filter)
    # ---------------------------------------------------------

    # Accumulate
    # ---------------------------------------------------------
    dft_filter_accum = np.zeros((height_dft, width_dft, 2), dtype=np.float32)

    # The value 12 was obtained experimentally in the visual cortex study.
    angle_step = 12

    for angle in range(0, 180, angle_step):

        if angle > 0:
            dft_filter_shift_rot = dft_rotate(
                                        dft_filter_shift,
                                        height_dft, width_dft,
                                        angle)

            dft_filter_accum[:, :, 0] += dft_filter_shift_rot[:, :, 0]
            dft_filter_accum[:, :, 1] += dft_filter_shift_rot[:, :, 1]
        else:
            dft_filter_accum[:, :, 0] += dft_filter_shift[:, :, 0]
            dft_filter_accum[:, :, 1] += dft_filter_shift[:, :, 1]
    # ---------------------------------------------------------

    return dft_filter_accum


def dft_rotate(dft, height_dft, width_dft, angle):

    re = dft[:, :, 0]
    im = dft[:, :, 1]

    mat = cv.getRotationMatrix2D((width_dft / 2, height_dft / 2), angle, 1)

    re_rot = cv.warpAffine(re, mat, (width_dft, height_dft))
    im_rot = cv.warpAffine(im, mat, (width_dft, height_dft))

    dft_rot = cv.merge([re_rot, im_rot])

    return dft_rot


def sobel_kernel(height_dft, width_dft):

    array_filter = np.zeros((height_dft, width_dft), dtype=np.float32)

    array_filter[0, 0] = 1
    array_filter[0, 1] = 2
    array_filter[0, 2] = 1

    array_filter[1, 0] = 0
    array_filter[1, 1] = 0
    array_filter[1, 2] = 0

    array_filter[2, 0] = -1
    array_filter[2, 1] = -2
    array_filter[2, 2] = -1

    return array_filter

