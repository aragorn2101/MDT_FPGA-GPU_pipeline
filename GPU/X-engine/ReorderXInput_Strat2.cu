/*
 *  GPU kernel to re-order data output from F-engine, to input to cuBLAS
 *  library routine (CUDA C)
 *  --  Strategy 2  --
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
 *  NpolsxNelements: number of elements in the array x number of polarisations
 *                   this flexibility allows to process for a single
 *                   polarisation if needed,
 *  Nchannels: number of frequency channels in each spectrum,
 *  FOutput: array output from F-engine,
 *  XInput: array to be input to cuBLAS kernel.
 *
 *  NumThreadx = (Npols*Nelements >= 32) ? 32 : (Npols*Nelements);
 *  NumThready = 32;
 *  NumThreadz = 1;
 *
 *  NumBlockx  = (Npols*Nelements)/NumThreadx + (((Npols*Nelements)%NumThreadx != 0) ? 1 : 0);
 *  NumBlocky  = Nchannels/NumThready + ((Nchannels%NumThready != 0) ? 1 : 0);
 *  NumBlockz  = Nspectra;
 *
 */
__global__ void ReorderXInput(int NpolsxNelements,
                              int Nchannels,
                              cufftComplex *FOutput,
                              cuComplex *XInput)
{
  __shared__ cufftComplex sh_Temp[32][33];
  int channelIdx, elementIdx;


  /*  Read data from output of F-engine  */
  channelIdx = blockIdx.y*blockDim.y + threadIdx.x;
  elementIdx = blockIdx.x*blockDim.x + threadIdx.y;

  if (channelIdx < Nchannels && elementIdx < NpolsxNelements)
    sh_Temp[threadIdx.x][threadIdx.y] = FOutput[ (blockIdx.z*NpolsxNelements + elementIdx)*Nchannels + channelIdx ];

  /*  Make sure that all data reads are completed before proceeding  */
  __syncthreads();

  /*  Write data to input array for X-engine  */
  channelIdx = channelIdx - threadIdx.x + threadIdx.y;
  elementIdx = elementIdx - threadIdx.y + threadIdx.x;

  if (channelIdx < Nchannels && elementIdx < NpolsxNelements)
    XInput[ (channelIdx*gridDim.z + blockIdx.z)*NpolsxNelements + elementIdx ] = sh_Temp[threadIdx.y][threadIdx.x];
}
