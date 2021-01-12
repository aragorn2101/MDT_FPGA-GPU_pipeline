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
