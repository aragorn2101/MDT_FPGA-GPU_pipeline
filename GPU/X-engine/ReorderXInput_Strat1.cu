/*
 *  GPU kernel to re-order data output from F-engine, to input to cuBLAS
 *  library routine (CUDA C)
 *  --  Strategy 1  --
 *
 *  Copyright (c) 2020 Nitish Ragoomundun
 *                     lrugratz gmail com
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


/*
 *  Nchannels: number of frequency channels in each spectrum,
 *  FOutput: array output from F-engine,
 *  XInput: array to be input to cuBLAS kernel.
 *
 *  NumThreadx = (Nchannels > MaxThreadsPerBlock) ? MaxThreadsPerBlock : Nchannels;
 *  NumThready = 1;
 *  NumThreadz = 1;
 *
 *  NumBlockx  = Nchannels/MaxThreadsPerBlock + ((Nchannels%MaxThreadsPerBlock != 0) ? 1 : 0);
 *  NumBlocky  = Nspectra;
 *  NumBlockz  = Npols*Nelements;
 *
 */
__global__ void ReorderXInput(int Nchannels, cufftComplex *FOutput, cuComplex *XInput)
{
  int F_Idx, X_Idx;

  int channelIdx = blockIdx.x*blockDim.x + threadIdx.x;

  // Input array arrangement from slowest varying index
  // to most rapidly varying:
  // Spectrum -> Element -> Pol -> Channel
  F_Idx = (blockIdx.y*gridDim.z + blockIdx.z)*Nchannels + channelIdx;

  // Output array arrangement from slowest varying index
  // to most rapidly varying:
  // Channel -> Spectrum -> Element -> Pol
  X_Idx = (channelIdx*gridDim.y + blockIdx.y)*gridDim.z + blockIdx.z;

  if (channelIdx < Nchannels)
    XInput[ X_Idx ] = FOutput[ F_Idx ];
}
