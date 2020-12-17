/*
 *  GPU kernel to compute polyphase structure (CUDA C)
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
 *  Nchannels: number of frequency channels in output spectra,
 *  Ntaps: number of taps for PFB,
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
                          cuComplex *InSignal,
                          cufftComplex *PolyStruct)
{
  // NOTE: input and output array arrangement from slowest varying index
  // to most rapidly varying:
  // Spectrum -> Element -> Pol -> Channel

  int i;
  int channelIdx = threadIdx.x + blockIdx.x*blockDim.x;
  float tmp1, tmp2, filter_coeff;
  cuComplex tmp_input;
  cufftComplex tmp_product;

  /*  Array position offset wrt element index  */
  long offset_elem = blockIdx.z * Nchannels;

  /*  Stride for successive spectra  */
  long stride_spec = gridDim.z * Nchannels;

  /*  Array position offset wrt spectrum index  */
  long offset_spec = blockIdx.y * stride_spec;


  tmp_product.x = 0.0f;
  tmp_product.y = 0.0f;

  for (i=0 ; i<Ntaps ; i++)
  {
    /*  Read input signal data and initialise tmp_product variable  */
    tmp_input  = InSignal[offset_spec + offset_elem + i*stride_spec + channelIdx];

    /***  BEGIN Calculate FIR filter  ***/

    tmp1 = (channelIdx + (i - 0.5f*Ntaps)*Nchannels) / Nchannels;
    if (tmp1 == 0.0f)  /*  To prevent division by 0  */
      filter_coeff = 1.0f ;
    else
      filter_coeff = sinpif(tmp1) / (d_PI * tmp1);

    tmp2 = 2.0f*(channelIdx + i*Nchannels) / (Ntaps*Nchannels);

    filter_coeff *= 0.35875f - 0.48829f*cospif(tmp2) + 0.14128f*cospif(2.0f*tmp2) - 0.01168f*cospif(3.0f*tmp2);

    /***  END Calculate FIR filter  ***/


    /*  Accumulate FIR  */
    tmp_product.x = fmaf(filter_coeff, tmp_input.x, tmp_product.x);  // I
    tmp_product.y = fmaf(filter_coeff, tmp_input.y, tmp_product.y);  // Q
  }

  /*  Write to output array  */
  PolyStruct[offset_spec + offset_elem + channelIdx] = tmp_product;
}
