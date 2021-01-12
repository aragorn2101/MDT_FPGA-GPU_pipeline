#include "MDT_device_functions.h"


/*
 *  GPU kernel to compute FIR filter coefficients.
 *  --  minimum 4-term Blackman-Harris window  --
 *
 *  Nchannels: number of frequency channels expected in PFB output spectrum,
 *  Ntaps: number of taps in polyphase filter bank,
 *  Window: array of size (Ntaps x Nchannels), in GPU global memory, to store
 *  coefficients of PFB FIR filter.
 *
 *  NumThreadx = MaxThreadsPerBlock;
 *  NumThready = 1;
 *  NumThreadz = 1;
 *  NumBlockx  = (Ntaps*Nchannels)/NumThreadx + (((Ntaps*Nchannels)%NumThreadx != 0) ? 1 : 0);
 *  NumBlocky  = 1;
 *  NumBlockz  = 1;
 *
 */
__global__ void PFB_Window_min4term_BH(int Nchannels,
                                       int Ntaps,
                                       float *Window)
{
  long idx = threadIdx.x + blockIdx.x*blockDim.x;

  /*  Temporary variables to prevent redundant computation  */
  float tmp1, tmp2, tmp_filter, tmp_window;

  if (idx < Ntaps*Nchannels)
  {

    /*
     *  Window: Sinc
     *
     *  sinc( ( channel - (Ntaps x Nchannels)/2 ) / Nchannels )
     *
     */
    tmp1 = (idx - 0.5f*Ntaps*Nchannels) / Nchannels;

    if ( tmp1 == 0.0f )  /*  To prevent division by 0  */
      tmp_filter = 1.0f ;
    else
      tmp_filter = sinpif( tmp1 ) / ( d_PI * tmp1 );


    /*
     *  Window: minimum 4-term Blackman-Harris
     */
    tmp2 = 2.0f*idx / (Ntaps*Nchannels);
    tmp_window = 0.35875f - 0.48829f*cospif(tmp2) + 0.14128f*cospif(2.0f*tmp2) - 0.01168f*cospif(3.0f*tmp2);


    /*  Write Windowed Sinc to global memory array  */
    Window[idx] = tmp_filter * tmp_window;
  }
}



/*
 *  GPU kernel to compute polyphase structure.
 *
 *  Nchannels: number of frequency channels in output spectra,
 *  Ntaps: number of taps for PFB,
 *  in GPU global memory:
 *  Window: array of size (Ntaps x Nchannels) containing filter coeff,
 *  InSignal: array of size (Nspectra - 1 + Ntaps) x Nchannels containing
 *            interleaved IQ samples of the input signal in an array of
 *            cufftComplex vectors (I:x, Q:y),
 *  PolyStruct: array of size Nchannels which will hold output.
 *
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
  cufftComplex tmp_input, tmp_product;

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
      tmp_input  = InSignal[offset_spec + offset_elem + i*stride_spec + channelIdx];
      tmp_window = Window[i*Nchannels + channelIdx];

      /*  Accumulate FIR  */
      tmp_product.x = fmaf(tmp_window, tmp_input.x, tmp_product.x);  // I
      tmp_product.y = fmaf(tmp_window, tmp_input.y, tmp_product.y);  // Q
    }

    // NOTE: in order to output specific order for use with Reorder kernel for cuBLAS:
    // Output array arrangement from slowest varying index
    // to most rapidly varying:
    // Spectrum -> Element -> Pol -> Channel

    PolyStruct[offset_spec + offset_elem + channelIdx] = tmp_product;
  }
}



/*
 *  GPU kernel to re-order data output from F-engine
 *
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
 *  NumBlockx  = (Npols*Nelements)/NumThreadx + (((Npols*Nelements)%NumThreadx != 0) ? 1 : 0);
 *  NumBlocky  = Nchannels/NumThready + ((Nchannels%NumThready != 0) ? 1 : 0);
 *  NumBlockz  = Nspectra;
 *
 */
__global__ void ReorderFOutput(int NpolsxNelements,
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
