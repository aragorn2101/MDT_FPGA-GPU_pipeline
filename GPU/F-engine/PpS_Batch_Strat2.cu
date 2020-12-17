/*
 *  GPU kernel to compute polyphase structure (CUDA C)
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
 *  Nchannels: number of frequency channels in output spectra,
 *  Ntaps: number of taps for PFB,
 *  Window: array of size (Ntaps x Nchannels) containing filter coeff,
 *  InSignal: array of size (Nspectra - 1 + Ntaps) x Nchannels containing
 *            interleaved IQ samples of the input signal in an array of
 *            cufftComplex vectors (I:x, Q:y),
 *  PolyStruct: array of size Nchannels which will hold output.
 *
 *  NumThreadx = MaxThreadsPerBlock
 *  NumThready = 1
 *  NumThreadz = 1
 *  NumBlockx  = Nchannels/NumThreadx + ((Nchannels%NumThreadx != 0) ? 1 : 0)
 *  NumBlocky  = Nspectra
 *  NumBlockz  = Npol*Nelements
 *
 */
__global__ void PpS_Batch(int Nchannels,
                          int Ntaps,
                          float *Window,
                          cuComplex *InSignal,
                          cufftComplex *PolyStruct)
{
  int i;
  int channelIdx = threadIdx.x + blockIdx.x*blockDim.x;
  float tmp_window;
  cuComplex tmp_input;
  cufftComplex tmp_product;

  // Input array arrangement from slowest varying index
  // to most rapidly varying:
  // Spectrum -> Element -> Pol -> Channel

  /*  Array position offset wrt element index  */
  long offset_elem = blockIdx.z * Nchannels;

  /*  Stride for successive spectra  */
  long stride_spec = gridDim.z * Nchannels;

  /*  Array position offset wrt spectrum index  */
  long offset_spec = blockIdx.y * stride_spec;


  if (channelIdx < Nchannels)
  {
    tmp_product.x = 0.0f;
    tmp_product.y = 0.0f;

    for (i=0 ; i<Ntaps ; i++)
    {
      /*  Read input signal data and filter coefficient  */
      // Input array arrangement from slowest varying index
      // to most rapidly varying:
      // Spectrum -> Element -> Pol -> Channel
      tmp_input  = InSignal[offset_spec + offset_elem + i*stride_spec + channelIdx];
      tmp_window = Window[i*Nchannels + channelIdx];

      /*  Accumulate FIR  */
      tmp_product.x = fmaf(tmp_window, tmp_input.x, tmp_product.x);  // I
      tmp_product.y = fmaf(tmp_window, tmp_input.y, tmp_product.y);  // Q
    }

    // Output array arrangement from slowest varying index
    // to most rapidly varying:
    // Spectrum -> Element -> Pol -> Channel
    PolyStruct[offset_spec + offset_elem + channelIdx] = tmp_product;
  }
}
