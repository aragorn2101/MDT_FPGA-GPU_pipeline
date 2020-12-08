/*
 *  GPU kernel to compute polyphase structure
 *  --  Strategy 3  --
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
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 *
 */

/*
 *  Nchannels: number of frequency channels in output spectra,
 *  in GPU global memory:
 *  InSignal: array of size (Nspectra - 1 + Ntaps) x Nchannels containing
 *            interleaved IQ samples of the input signal in an array of
 *            cufftComplex vectors (I:x, Q:y),
 *  PolyStruct: array of size Nchannels which will hold output.
 *  sh_Product[]: shared memory space must be allocated in kernel call,
 *                amount of shared memory must be
 *                sizeof(cufftComplex) x Ntaps x NumThreadx
 *
 *  NumThreadx = (MaxThreadsPerBlock * 2) / Ntaps
 *  NumThready = Ntaps / 2
 *  NumThreadz = 1
 *  NumBlockx  = Nchannels/NumThreadx + ((Nchannels%NumThreadx != 0) ? 1 : 0)
 *  NumBlocky  = Nspectra
 *  NumBlockz  = Npol*Nelements
 *
 */
__global__ void PpS_Batch(int Nchannels,
                          float *Window,
                          cuComplex *InSignal,
                          cufftComplex *PolyStruct)
{
  extern __shared__ cufftComplex sh_Product[];

  int i;
  float tmp_window;
  cuComplex tmp_input;
  cufftComplex tmp_product;

  /*  Variables for strides to eliminate redundant calculation  */
  int stride1 = threadIdx.y*blockDim.x + threadIdx.x,
      stride2 =  blockIdx.x*blockDim.x + threadIdx.y*Nchannels + threadIdx.x,
      stride3 =  blockIdx.x*blockDim.x + threadIdx.y*Nchannels + blockIdx.y*Nchannels + threadIdx.x;


  /***  Read filter into register, read input data, multiply and store in shared memory  ***/

  /*  First Ntaps/2 segments  */
  tmp_window = Window[stride2];
  tmp_input = InSignal[stride3];
  tmp_product.x = tmp_window * tmp_input.x;
  tmp_product.y = tmp_window * tmp_input.y;

  sh_Product[stride1] = tmp_product;

  /*  Second Ntaps/2 segments  */
  tmp_window = Window[stride2 + blockDim.y*Nchannels];
  tmp_input = InSignal[stride3 + blockDim.y*Nchannels];
  tmp_product.x = tmp_window * tmp_input.x;
  tmp_product.y = tmp_window * tmp_input.y;

  sh_Product[stride1 + blockDim.y*blockDim.x] = tmp_product;

  __syncthreads();


  /***  Loop through shared memory to sum reduce  ***/
  for (i=blockDim.y ; i>0 ; i>>=1)
  {
    if (threadIdx.y < i)
    {
      sh_Product[stride1].x += sh_Product[i*blockDim.x + stride1].x;
      sh_Product[stride1].y += sh_Product[i*blockDim.x + stride1].y;
    }
    __syncthreads();
  }


  // NOTE: in order to output specific order for use with Reorder kernel for cuBLAS:
  // Output array arrangement from slowest varying index
  // to most rapidly varying:
  // Spectrum -> Element -> Pol -> Channel
  if (threadIdx.y == 0)  // only blockDim.x values to write
    PolyStruct[(blockIdx.z*blockDim.y + blockIdx.y)*Nchannels + blockIdx.x*blockDim.x + threadIdx.x] = sh_Product[threadIdx.x];
}
