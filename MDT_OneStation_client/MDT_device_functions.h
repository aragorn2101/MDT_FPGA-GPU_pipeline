/*
 *  MDT_device_functions.h
 *
 *  Header file for the kernel source file.
 *
 *  Copyright (c) 2021 Nitish Ragoomundun
 *                    <lrugratz gmail com>
 *                             @     .
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a
 *  copy of this software and associated documentation files (the "Software"),
 *  to deal in the Software without restriction, including without limitation
 *  the rights to use, copy, modify, merge, publish, distribute, sublicense,
 *  and/or sell copies of the Software, and to permit persons to whom the
 *  Software is furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *  DEALINGS IN THE SOFTWARE.
 *
 */


#ifndef MDT_DEVICE_FUNC
#define MDT_DEVICE_FUNC

#include <cuda.h>
#include <cufft.h>
#include <cublas_v2.h>


/*  Constants for device  */
__device__ __constant__ float d_PI = 3.141592653589793238462643383279502884f;


/*  Device kernel prototypes  */
__global__ void PFB_Window_min4term_BH(int, int, float *);
__global__ void PpS_Batch(int, int, float *, cuComplex *, cufftComplex *);
__global__ void ReorderFOutput(int, int, cufftComplex *, cuComplex *);

#endif
