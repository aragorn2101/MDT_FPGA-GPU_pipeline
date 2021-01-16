#include "MDT_OneStation_client.h"

int main (int argc, char **argv)
{
  /***  BEGIN Parse command line arguments  ***/

  /*  Check number of command line arguments  */
  if (argc <= 5) // argc <= (number of expected arguments)
    print_help(1, argv[0]);


  /*  Parameters for telescope  */
  int Nelements, Npols, Nchannels, Ntaps, Nspectra;

  /*  Initialise parameters from command line arguments  */
  /*  Number of elements in the instrument  */
  Nelements = atoi(argv[1]);
  if (Nelements < MINELEMENTS || Nelements > MAXELEMENTS)
  {
    printf("%s: Invalid number of elements!\n", argv[0]);
    print_help(2, argv[0]);
  }

  /*  Number of polarisations for each element  */
  Npols = atoi(argv[2]);
  if (Npols < 1 || Npols > 2)
  {
    printf("%s: Invalid number of polarisations!\n", argv[0]);
    print_help(3, argv[0]);
  }

  /*  Number of frequency channels in each spectrum  */
  Nchannels = atoi(argv[3]);
  if (Nchannels < MINCHANNELS || Nchannels > MAXCHANNELS)
  {
    printf("%s: Invalid number of frequency channels!\n", argv[0]);
    print_help(4, argv[0]);
  }

  /*  Number of taps in PFB  */
  Ntaps = atoi(argv[4]);
  if (Ntaps < MINTAPS || Ntaps > MAXTAPS)
  {
    printf("%s: Invalid number of taps for PFB!\n", argv[0]);
    print_help(5, argv[0]);
  }

  /*  Number of spectra for 1 integration time  */
  Nspectra = atoi(argv[5]);
  if (Nspectra < MINSPECTRA || Nspectra > MAXSPECTRA)
  {
    printf("%s: Invalid number of spectra !\n", argv[0]);
    print_help(6, argv[0]);
  }

  /***  END Parse command line arguments  ***/



  /***  BEGIN: Simulation signal characteristics  ***/

  /*  Initialise Gaussian RNG on host  */
  GaussSeed(time(0));

  /*  Sampling frequency  */
  float Fs = 250e3;  // 250 kSps

  /*  delta t corresponding to sampling frequency  */
  double dt = 1.0 / Fs;
  double t;
  double t_global = 0.0;

  /*  Frequencies  */
  double nu1 =  82.0 * Fs / Nchannels;  // bin  82: 327.295 MHz
  double nu2 = 307.0 * Fs / Nchannels;  // bin 307: 327.350 MHz
  double nu3 = 598.0 * Fs / Nchannels;  // bin 598: 327.421 MHz
  double nu4 = 921.0 * Fs / Nchannels;  // bin 921: 327.500 MHz

  /*  Amplitudes  */
  float A1 = 0.50, A2 = 1.00, A3 = 0.25, A4 = 0.75;

  /*  Initial Phases  */
  float phi1 = 0.125*PI, phi2 = 0.430*PI, phi3 = 0.200*PI, phi4 = 0.322*PI;

  /*  Maximum noise amplitude  */
  float NoiseAmp = 10.0;

  /*  Phase jitter amplitude  */
  float PhsJitterAmp = 0.5*PI;

  /*  Geometrical offset phase due to antenna's position  */
  float geo_phs;

  /***  END: Simulation signal characteristics  ***/



  /***  BEGIN Variables for array lengths in memory  ***/

  /*  Size of window for PFB  */
  long NBytes_Window;

  /*  Size of stack for input samples  */
  long NBytes_Stack;

  /*  Stride after the first (Ntaps - 1) segments of samples  */
  long stride_Ntaps;

  /*  Stride after the first Nspectra segments of samples  */
  long stride_Nspectra;

  /*  Size of data copied from host to GPU at input  */
  long NBytes_HtoD;

  /*  Size of input array to F-engine  */
  long NBytes_FInput;

  /*  Size of data to shift in device memory between successive PFB's  */
  long NBytes_DtoD;

  /*  Size of output array from F-engine  */
  long NBytes_FOutput;

  /*  Size of input array to the GPU X-engine  */
  long NBytes_XInput;

  /*  Size of cuBLAS pointer arrays  */
  long NBytes_cuBLASptr_Xin;
  long NBytes_cuBLASptr_Xout;

  /*  Size of output array copied from GPU to host at output  */
  /*  Size of X-engine output array  */
  long NBytes_DtoH;

  /*  stride to use with column-major correlation matrix  */
  long stride_cublas;

  /***  END Variables for array lengths in memory  ***/



  /***  BEGIN Host array sizes  ***/

  /*  Size of window for PFB  */
  NBytes_Window = sizeof(float) * Ntaps * Nchannels;

  /*  Size of stack for input samples  */
  NBytes_Stack = sizeof(cuComplex) * Npols*Nelements * Nchannels * Nspectra;


  /***  END Host array sizes  ***/



  /***  BEGIN Device array sizes  ***/

  /*  Size of input array to F-engine  */
  NBytes_FInput = sizeof(cuComplex) * Npols*Nelements * Nchannels * (Nspectra + Ntaps - 1);

  /*  Stride after the first (Ntaps - 1) segments of samples  */
  stride_Ntaps = Npols*Nelements * Nchannels * (Ntaps - 1);

  /*  Stride after the first Nspectra segments of samples  */
  stride_Nspectra = Npols*Nelements * Nchannels * Nspectra;

  /*  Size of output array from F-engine  */
  NBytes_FOutput = sizeof(cufftComplex) * Npols*Nelements * Nchannels * Nspectra;

  /*  Size of input array to the GPU X-engine  */
  NBytes_XInput = sizeof(cuComplex) * Npols*Nelements * Nchannels * Nspectra;

  /*  Size of cuBLAS pointer arrays  */
  NBytes_cuBLASptr_Xin  = sizeof(cuComplex *) * Nchannels;
  NBytes_cuBLASptr_Xout = sizeof(cuComplex *) * Nchannels;

  /***  END Device array sizes  ***/



  /***  BEGIN Data transfer sizes  ***/

  /*  Size of data copied from host to device at input  */
  NBytes_HtoD = sizeof(cuComplex) * Npols*Nelements * Nchannels * Nspectra;
  // input

  /*  Size of data to shift in device memory between successive PFB's  */
  NBytes_DtoD = sizeof(cuComplex) * Npols*Nelements * Nchannels * (Ntaps - 1);

  /*  Size of output data copied from device to host at output  */
  /*  same as size of correlation matrix output array  */
  NBytes_DtoH = sizeof(cuComplex) * Npols*Nelements * Npols*Nelements * Nchannels;
  // output

  /***  END Data transfer sizes  ***/



  /***  BEGIN Host variables  ***/

  /*  Array for unfiltered spectrum  */
  cufftComplex *h_Unfiltered;

  /*  Array for filtered spectrum  */
  cufftComplex *h_Filtered;

  /*  Stack for incoming signal data  */
  cuComplex *h_Stack;

  /*  Pointer arrays for use with cuBLAS X-engine  */
  cuComplex **h_XInputPtr, **h_XOutputPtr;

  /*  cuBLAS scalar coefficients  */
  cuComplex cublas_alpha, cublas_beta;

  /*  Output array for GPU FX correlator (pinned)  */
  cuComplex *h_XOutput;

  long int i, j, k;
  unsigned short int p, q;
  long int stride;

  /*  Variables for output file  */
  FILE *outfile;
  char outfilename[OUTFILENAMELEN];
  struct tm *now;
  time_t now_abs;

  /***  END Host variables  ***/



  /***  BEGIN Device array variables  ***/

  /*  Array for nfiltered spectrum  */
  cufftComplex *d_Unfiltered;

  /*  Array for window  */
  float *d_Window;

  /*  Input array for signal data  */
  cuComplex *d_FInput;

  /*  Output array from GPU PFB  */
  cufftComplex *d_FOutput;

  /*  Input array for X-engine  */
  cuComplex *d_XInput;

  /*  Pointer arrays for use with cuBLAS  */
  cuComplex **d_XInputPtr, **d_XOutputPtr;

  /*  Output array for X-engine  */
  cuComplex *d_XOutput;

  /***  END Device array variables  ***/



  /***  BEGIN CUDA control variables  ***/

  /*  GPU thread gridding parameters for device function PpS_Batch()  */
  int PpS_Batch_thdx, PpS_Batch_thdy, PpS_Batch_thdz;  // block dim
  int PpS_Batch_blkx, PpS_Batch_blky, PpS_Batch_blkz;  // grid dim

  /*  GPU thread gridding parameters for device function ReorderFOutput()  */
  int ReorderFOutput_thdx, ReorderFOutput_thdy, ReorderFOutput_thdz;  // block dim
  int ReorderFOutput_blkx, ReorderFOutput_blky, ReorderFOutput_blkz;  // block dim

  /*  General purpose return value  */
  int cu_RetVal;

  /*  CUDA Streams  */
  cudaStream_t cu_Stream[3];

  /*  cuFFT plan handle  */
  cufftHandle cu_FFT_Plan, tmp_FFT_Plan;

  /*  cuBLAS control  */
  cublasHandle_t cu_BLAS_XEngine;
  cublasStatus_t cu_BLAS_Stat;

  /***  END CUDA control variables  ***/



  /***  BEGIN Retrieving device properties and memory check  ***/
  int cu_devIdx;
  cudaDeviceProp cu_devProp;
  char *cu_DeviceName;
  int cuFFT_version;
  int cuBLAS_version_maj, cuBLAS_version_min, cuBLAS_version_patch;
  long long cu_MaxGPUGlobalMem;
  int cu_MaxThreadsPerBlk;

  cu_RetVal = cudaGetDevice(&cu_devIdx);
  if ( (cu_RetVal = cudaErrCheck()) != 0 ) exit(cu_RetVal);

  cu_RetVal = cudaGetDeviceProperties(&cu_devProp, cu_devIdx);
  if ( (cu_RetVal = cudaErrCheck()) != 0 ) exit(cu_RetVal);

  /*  Name of device  */
  cu_DeviceName = cu_devProp.name;

  /*  Get cuFFT version  */
  cufftGetVersion(&cuFFT_version);

  /*  Get cuBLAS version  */
  cublasGetProperty(MAJOR_VERSION, &cuBLAS_version_maj);
  cublasGetProperty(MINOR_VERSION, &cuBLAS_version_min);
  cublasGetProperty(PATCH_LEVEL, &cuBLAS_version_patch);

  /*  Maximum amount of global memory on device  */
  cu_MaxGPUGlobalMem = cu_devProp.totalGlobalMem;

  /*  Maximum number of threads per block  */
  cu_MaxThreadsPerBlk = cu_devProp.maxThreadsPerBlock;

  /***  END Retrieving device properties and memory check  ***/



  /***  BEGIN Check available memory size  ***/

  long long hostMemRequired = NBytes_Stack + NBytes_HtoD + NBytes_cuBLASptr_Xin + NBytes_cuBLASptr_Xout + NBytes_DtoH;
  long long devMemRequired  = NBytes_Window + NBytes_FInput + NBytes_FOutput + NBytes_XInput + NBytes_cuBLASptr_Xin + NBytes_cuBLASptr_Xout + NBytes_DtoH;

  if (hostMemRequired >= MAXHOSTRAM)
  {
    printf("Size of data arrays exceed available CPU memory space on\n");
    printf("device %s for the requested instrument configuration!\n", cu_DeviceName);
    exit(7);
  }

  if (devMemRequired >= cu_MaxGPUGlobalMem)
  {
    printf("Size of data arrays exceed available memory space\n");
    printf("for the requested instrument configuration!\n");
    exit(8);
  }

  /***  END Check available memory size  ***/



  printf("\n");
  printf("**************************************************\n");
  printf("                  Mode: Stand-by                  \n");
  printf("**************************************************\n\n");

  printf("Using %s (ID: %d) as device.\n", cu_DeviceName, cu_devIdx);
  printf("GPU F-engine is powered by cuFFT version %d.\n", cuFFT_version);
  printf("GPU X-engine is powered by cuBLAS version %d.%d-%d.\n", cuBLAS_version_maj, cuBLAS_version_min, cuBLAS_version_patch);
  printf("\n");

  printf("Allocating arrays on host and device ...\n");


  /***  BEGIN Host: Allocate arrays  ***/

  /*  Array for unfiltered spectrum  */
  h_Unfiltered = (cufftComplex *)malloc(sizeof(cufftComplex) * Npols*Nelements * Nchannels);

  /*  Array for filtered spectrum  */
  h_Filtered = (cufftComplex *)malloc(sizeof(cufftComplex) * Npols*Nelements * Nchannels);

  /*  Stack for incoming signal data (page-locked)  */
  cudaHostAlloc((void **)&h_Stack, NBytes_Stack, cudaHostAllocDefault);

  /*  Pointer arrays for use with cuBLAS X-engine  */
  h_XInputPtr  = (cuComplex **)malloc(NBytes_cuBLASptr_Xin);
  h_XOutputPtr = (cuComplex **)malloc(NBytes_cuBLASptr_Xout);

  /*  Output array in host memory  */
  cudaHostAlloc((void **)&h_XOutput, NBytes_DtoH, cudaHostAllocDefault);

  /***  END Host: Allocate arrays  ***/


  /***  BEGIN Device: Allocate arrays  ***/

  /*  Array for unfiltered spectrum  */
  cudaMalloc((void **)&d_Unfiltered, sizeof(cufftComplex) * Npols*Nelements * Nchannels);

  /*  Array for window  */
  cudaMalloc((void **)&d_Window, NBytes_Window);

  /*  Input array to F-engine  */
  cudaMalloc((void **)&d_FInput, NBytes_FInput);

  /*  Output array from F-engine  */
  cudaMalloc((void **)&d_FOutput, NBytes_FOutput);

  /*  Input array for X-engine  */
  cudaMalloc((void **)&d_XInput, NBytes_XInput);

  /*  Pointer arrays for use with cuBLAS  */
  cudaMalloc((void ***)&d_XInputPtr,  NBytes_cuBLASptr_Xin);
  cudaMalloc((void ***)&d_XOutputPtr, NBytes_cuBLASptr_Xout);

  /*  Output array for X-engine  */
  cudaMalloc((void **)&d_XOutput, NBytes_DtoH);

  printf("Done allocating arrays.\n");

  /***  END Device: Allocate arrays  ***/



  /***  BEGIN Device: Create CUDA streams  ***/

  for (i=0 ; i<3 ; i++)  // create 3 streams
    cudaStreamCreate(&cu_Stream[i]);

  /***  END Device: Create CUDA streams  ***/



  /***  BEGIN Device: Initialise window coefficients for PFB  ***/

  printf("\nInitialising window coefficients for PFB ...\n");
  printf("Grid configuration  = (%6d,%6d,%6d)\n", cu_MaxThreadsPerBlk, 1, 1);
  printf("Block configuration = (%6d,%6d,%6d)\n", (Ntaps*Nchannels)/cu_MaxThreadsPerBlk + (((Ntaps*Nchannels)%cu_MaxThreadsPerBlk != 0) ? 1 : 0), 1, 1);
  PFB_Window_min4term_BH<<< cu_MaxThreadsPerBlk, (Ntaps*Nchannels)/cu_MaxThreadsPerBlk + (((Ntaps*Nchannels)%cu_MaxThreadsPerBlk != 0) ? 1 : 0), 0, cu_Stream[0] >>>(Nchannels, Ntaps, d_Window);
  cudaStreamSynchronize(cu_Stream[0]);
  if ( (cu_RetVal = cudaErrCheck()) != 0 ) exit(cu_RetVal);
  printf("Done.\n");

  /***  END Device: Initialise window coefficients for PFB  ***/



  /***  BEGIN Device: PpS_Batch() grid characteristics  ***/

  printf("\nComputing GPU grid structure for  the PpS_Batch() kernel ...\n");
  PpS_Batch_thdx = cu_MaxThreadsPerBlk;
  PpS_Batch_thdy = 1;
  PpS_Batch_thdz = 1;

  PpS_Batch_blkx = Nchannels/PpS_Batch_thdx + ((Nchannels%PpS_Batch_thdx != 0) ? 1 : 0);
  PpS_Batch_blky = Nspectra;
  PpS_Batch_blkz = Npols*Nelements;
  printf("Grid configuration  = (%6d,%6d,%6d)\n", PpS_Batch_blkx, PpS_Batch_blky, PpS_Batch_blkz);
  printf("Block configuration = (%6d,%6d,%6d)\n", PpS_Batch_thdx, PpS_Batch_thdy, PpS_Batch_thdz);
  printf("Done.\n");

  /***  END Device: PpS_Batch() grid characteristics  ***/



  /***  BEGIN Device: prepare cuFFT plan  ***/

  printf("\nConstructing cuFFT plan prior to processing ...\n");
  if ((cu_RetVal = cufftPlan1d(&cu_FFT_Plan, Nchannels, CUFFT_C2C, Npols*Nelements*Nspectra)) != CUFFT_SUCCESS)
  {
    fprintf(stderr, "CUFFT error (%d): failed to create FFT plan for F-engine.\n", cu_RetVal);
    exit(cu_RetVal);
  }

  printf("Setting up stream for cuFFT ...\n");
  cufftSetStream(cu_FFT_Plan, cu_Stream[1]);
  if ( (cu_RetVal = cudaErrCheck()) != 0 ) exit(cu_RetVal);
  printf("Done.\n");

  /***  END Device: prepare cuFFT plan  ***/



  /***  BEGIN Device: ReorderFOutput() grid characteristics  ***/

  printf("\nComputing GPU grid structure for  the ReorderFOutput() kernel ...\n");
  ReorderFOutput_thdx = 32;
  ReorderFOutput_thdy = 32;
  ReorderFOutput_thdz = 1;

  ReorderFOutput_blkx = (Npols*Nelements)/ReorderFOutput_thdx + (((Npols*Nelements)%ReorderFOutput_thdx != 0) ? 1 : 0);
  ReorderFOutput_blky = Nchannels/ReorderFOutput_thdy + ((Nchannels%ReorderFOutput_thdy != 0) ? 1 : 0);
  ReorderFOutput_blkz = Nspectra;
  printf("Grid configuration  = (%6d,%6d,%6d)\n", ReorderFOutput_blkx, ReorderFOutput_blky, ReorderFOutput_blkz);
  printf("Block configuration = (%6d,%6d,%6d)\n", ReorderFOutput_thdx, ReorderFOutput_thdy, ReorderFOutput_thdz);
  printf("Done.\n");

  /***  END Device: ReorderFOutput() grid characteristics  ***/



  /***  BEGIN Device: Initialise cuBLAS  ***/

  printf("\nInitialising pointer arrays for use with cuBLAS ...\n");
  for (i=0 ; i<Nchannels ; i++)
  {
    h_XInputPtr[i]  = d_XInput  + i*Npols*Nelements*Nspectra;
    h_XOutputPtr[i] = d_XOutput + i*Npols*Nelements*Npols*Nelements;
  }

  /*  Copy pointer arrays to device  */
  printf("Copying cuBLAS pointer arrays to device ...\n");
  cudaMemcpy((void **)d_XInputPtr, (void **)h_XInputPtr, NBytes_cuBLASptr_Xin, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  if ( (cu_RetVal = cudaErrCheck()) != 0 ) exit(cu_RetVal);

  cudaMemcpy((void **)d_XOutputPtr, (void **)h_XOutputPtr, NBytes_cuBLASptr_Xout, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  if ( (cu_RetVal = cudaErrCheck()) != 0 ) exit(cu_RetVal);

  free(h_XInputPtr);
  free(h_XOutputPtr);

  /*  Initialise scalar coefficients and set stream  */
  printf("\nInitialise cuBLAS scalar coefficients and X-engine stream ...\n");
  cublas_alpha.x = 1.0f / Nspectra;
  cublas_alpha.y = 0.0f;
  cublas_beta.x  = 0.0f;
  cublas_beta.y  = 0.0f;

  cu_BLAS_Stat = cublasCreate(&cu_BLAS_XEngine);
  if ( (cu_RetVal = cudaErrCheck()) != 0 ) exit(cu_RetVal);

  cu_BLAS_Stat = cublasSetStream(cu_BLAS_XEngine, cu_Stream[1]);
  if ( (cu_RetVal = cudaErrCheck()) != 0 ) exit(cu_RetVal);

  printf("Done.\n");

  /***  END Device: Initialise cuBLAS  ***/



  printf("\n");
  printf("**************************************************\n");
  printf("                  Mode: Observing                 \n");
  printf("**************************************************\n\n");



  /***  BEGIN Host: Generate simulated test signal  ***/
  /***  (Ntaps - 1) segments  ***/

  printf("Generating first (Ntaps - 1) segments of test signal ...\n");

  // Arrangement from slowest varying index to most rapidly varying
  // Spectrum -> Element -> Pol -> Channel

  for (i=0 ; i<(Ntaps - 1) ; i++)  // loop over spectra
  {
    for (j=0 ; j<Nelements ; j++)  // loop over elements
    {
      geo_phs =  j * (PI / 10);

      for (p=0 ; p<Npols ; p++)  // loop over polarisations
      {
        t = t_global;
        stride = (i*Npols*Nelements + j*Npols + p) * Nchannels;


        for (k=0 ; k<Nchannels ; k++)  // loop over channels
        {
          // I
          h_Stack[stride + k].x = A1*cos(2.0*PI*nu1*t + phi1 + PhsJitterAmp*GaussRand(0.0, 0.622222) + geo_phs)
                                + A2*cos(2.0*PI*nu2*t + phi2 + PhsJitterAmp*GaussRand(0.0, 0.622222) + geo_phs)
                                + A3*cos(2.0*PI*nu3*t + phi3 + PhsJitterAmp*GaussRand(0.0, 0.622222) + geo_phs)
                                + A4*cos(2.0*PI*nu4*t + phi4 + PhsJitterAmp*GaussRand(0.0, 0.622222) + geo_phs)
                                + NoiseAmp*GaussRand(0.0, 0.622222);

          // Q
          h_Stack[stride + k].y = A1*cos(2.0*PI*nu1*t + phi1 + PhsJitterAmp*GaussRand(0.0, 0.622222) + geo_phs - 0.5*PI)
                                + A2*cos(2.0*PI*nu2*t + phi2 + PhsJitterAmp*GaussRand(0.0, 0.622222) + geo_phs - 0.5*PI)
                                + A3*cos(2.0*PI*nu3*t + phi3 + PhsJitterAmp*GaussRand(0.0, 0.622222) + geo_phs - 0.5*PI)
                                + A4*cos(2.0*PI*nu4*t + phi4 + PhsJitterAmp*GaussRand(0.0, 0.622222) + geo_phs - 0.5*PI)
                                + NoiseAmp*GaussRand(0.0, 0.622222);

          t += dt;
        }
      }
    }

    t_global = t;
  }

  /***  END Host: Generate simulated test signal  ***/



  /***  BEGIN Copy (Ntaps - 1) segments of signal to device  ***/

  printf("Copying first (Ntaps - 1) segments of test signal to device ...\n");
  cudaMemcpyAsync((void *)d_FInput, (void *)h_Stack, NBytes_DtoD, cudaMemcpyHostToDevice, cu_Stream[0]);
  cudaStreamSynchronize(cu_Stream[0]);
  if ( (cu_RetVal = cudaErrCheck()) != 0 ) exit(cu_RetVal);
  printf("Done.\n");

  /***  END Copy (Ntaps - 1) segments of signal to device  ***/



  /***  BEGIN Host: Generate simulated test signal  ***/
  /***  Nspectra segments  ***/

  printf("Generating the next Nspectra segments of test signal ...\n");

  // Arrangement from slowest varying index to most rapidly varying
  // Spectrum -> Element -> Pol -> Channel

  for (i=0 ; i<Nspectra ; i++)  // loop over spectra
  {
    for (j=0 ; j<Nelements ; j++)  // loop over elements
    {
      geo_phs =  j * (PI / 10);

      for (p=0 ; p<Npols ; p++)  // loop over polarisations
      {
        t = t_global;
        stride = (i*Npols*Nelements + j*Npols + p) * Nchannels;


        for (k=0 ; k<Nchannels ; k++)  // loop over channels
        {
          // I
          h_Stack[stride + k].x = A1*cos(2.0*PI*nu1*t + phi1 + PhsJitterAmp*GaussRand(0.0, 0.622222) + geo_phs)
                                + A2*cos(2.0*PI*nu2*t + phi2 + PhsJitterAmp*GaussRand(0.0, 0.622222) + geo_phs)
                                + A3*cos(2.0*PI*nu3*t + phi3 + PhsJitterAmp*GaussRand(0.0, 0.622222) + geo_phs)
                                + A4*cos(2.0*PI*nu4*t + phi4 + PhsJitterAmp*GaussRand(0.0, 0.622222) + geo_phs)
                                + NoiseAmp*GaussRand(0.0, 0.622222);

          // Q
          h_Stack[stride + k].y = A1*cos(2.0*PI*nu1*t + phi1 + PhsJitterAmp*GaussRand(0.0, 0.622222) + geo_phs - 0.5*PI)
                                + A2*cos(2.0*PI*nu2*t + phi2 + PhsJitterAmp*GaussRand(0.0, 0.622222) + geo_phs - 0.5*PI)
                                + A3*cos(2.0*PI*nu3*t + phi3 + PhsJitterAmp*GaussRand(0.0, 0.622222) + geo_phs - 0.5*PI)
                                + A4*cos(2.0*PI*nu4*t + phi4 + PhsJitterAmp*GaussRand(0.0, 0.622222) + geo_phs - 0.5*PI)
                                + NoiseAmp*GaussRand(0.0, 0.622222);

          t += dt;
        }
      }
    }

    t_global = t;
  }

  /***  END Host: Generate simulated test signal  ***/



  /***  BEGIN Copy (Ntaps - 1) segments of signal to device  ***/

  printf("Copying the next Nspectra segments of test signal to device ...\n");
  cudaMemcpyAsync((void *)&d_FInput[stride_Ntaps], (void *)h_Stack, NBytes_HtoD, cudaMemcpyHostToDevice, cu_Stream[0]);
  cudaStreamSynchronize(cu_Stream[0]);
  if ( (cu_RetVal = cudaErrCheck()) != 0 ) exit(cu_RetVal);
  printf("Done.\n");

  /***  END Copy (Ntaps - 1) segments of signal to device  ***/



  /***  BEGIN TEST: unfiltered spectrum  ***/

  printf("Generate unfiltered spectrum and transfer results to host memory ...\n");

  /*  Copy a segment of Npols*Nelements*Nchannels of data  */
  cudaMemcpyAsync((void *)d_Unfiltered, (void *)d_FInput, sizeof(cufftComplex) * Npols*Nelements * Nchannels, cudaMemcpyDeviceToDevice, cu_Stream[2]);
  cudaStreamSynchronize(cu_Stream[2]);
  if ( (cu_RetVal = cudaErrCheck()) != 0 ) exit(cu_RetVal);

  /*  Prepare cuFFT plan  */
  if ((cu_RetVal = cufftPlan1d(&tmp_FFT_Plan, Nchannels, CUFFT_C2C, Npols*Nelements)) != CUFFT_SUCCESS)
  {
    fprintf(stderr, "CUFFT error (%d): failed to create FFT plan for F-engine.\n", cu_RetVal);
    exit(cu_RetVal);
  }

  cufftSetStream(tmp_FFT_Plan, cu_Stream[2]);
  if ( (cu_RetVal = cudaErrCheck()) != 0 ) exit(cu_RetVal);

  /*  Call cufftExecC2C()  */
  cufftExecC2C(tmp_FFT_Plan, d_Unfiltered, d_Unfiltered, CUFFT_FORWARD);
  cudaStreamSynchronize(cu_Stream[2]);
  if ( (cu_RetVal = cudaErrCheck()) != 0 ) exit(cu_RetVal);

  /*  Copy results to host memory  */
  cudaMemcpy((void *)h_Unfiltered, (void *)d_Unfiltered, sizeof(cufftComplex) * Npols*Nelements * Nchannels, cudaMemcpyDeviceToHost);
  printf("Done.\n");

  /***  END TEST: unfiltered spectrum  ***/



  /***  BEGIN Device: F-engine  ***/

  printf("Launching F-engine ...\n");

  /*  Compute polyphase structure  */
  printf("Calling PpS_Batch ...\n");
  PpS_Batch<<< dim3(PpS_Batch_blkx,PpS_Batch_blky,PpS_Batch_blkz), dim3(PpS_Batch_thdx,PpS_Batch_thdy,PpS_Batch_thdz), 0, cu_Stream[1] >>>(Nchannels, Ntaps, d_Window, d_FInput, d_FOutput);
  cudaStreamSynchronize(cu_Stream[1]);
  if ( (cu_RetVal = cudaErrCheck()) != 0 ) exit(cu_RetVal);

  /*  Copy last (Ntaps - 1) spectra to the beginning of input array to PFB  */
  printf("Copying last (Ntaps - 1) spectra to the beginning of input array to PFB ...\n");
  cudaMemcpyAsync((void *)d_FInput, (void *)&d_FInput[stride_Nspectra], NBytes_DtoD, cudaMemcpyDeviceToDevice, cu_Stream[2]);

  /*  In-place Fast Fourier Transform  */
  printf("Calling cufftExecC2C ...\n");
  cufftExecC2C(cu_FFT_Plan, d_FOutput, d_FOutput, CUFFT_FORWARD);
  cudaStreamSynchronize(cu_Stream[1]);
  if ( (cu_RetVal = cudaErrCheck()) != 0 ) exit(cu_RetVal);
  printf("Done.\n");

  /***  END Device: F-engine  ***/



  /***  BEGIN TEST: filtered spectrum  ***/

  printf("Taking a sample of filtered spectra ...\n");
  cudaMemcpy((void *)h_Filtered, (void *)d_FOutput, sizeof(cufftComplex) * Npols*Nelements * Nchannels, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  if ( (cu_RetVal = cudaErrCheck()) != 0 ) exit(cu_RetVal);
  printf("Done.\n");

  /***  END TEST: filtered spectrum  ***/



  /***  BEGIN Device: X-engine  ***/

  printf("Launching X-engine ...\n");

  /*  Re-order array for input to cuBLAS function  */
  ReorderFOutput<<< dim3(ReorderFOutput_blkx,ReorderFOutput_blky,ReorderFOutput_blkz), dim3(ReorderFOutput_thdx,ReorderFOutput_thdy,ReorderFOutput_thdz), 0, cu_Stream[1] >>>(Npols*Nelements, Nchannels, d_FOutput, d_XInput);
  cudaStreamSynchronize(cu_Stream[1]);
  if ( (cu_RetVal = cudaErrCheck()) != 0 ) exit(cu_RetVal);

  /*  Cross-Correlation engine using cuBLAS  */
  cu_BLAS_Stat = CUBLAS_STATUS_EXECUTION_FAILED;
  cu_BLAS_Stat = cublasCgemmBatched(cu_BLAS_XEngine, CUBLAS_OP_N, CUBLAS_OP_C, Npols*Nelements, Npols*Nelements, Nspectra, &cublas_alpha, (const cuComplex **)d_XInputPtr, Npols*Nelements, (const cuComplex **)d_XInputPtr, Npols*Nelements, &cublas_beta, d_XOutputPtr, Npols*Nelements, Nchannels);
  cudaStreamSynchronize(cu_Stream[1]);
  if (cu_BLAS_Stat != CUBLAS_STATUS_SUCCESS)
  {
    printf("\ncuBLAS X-engine failed (cublasStat = %d)!\n", (int)cu_BLAS_Stat);
    exit(9);
  }

  /***  END Device: X-engine  ***/



  /***  BEGIN Copy output correlation matrix to host memory  ***/

  /*  Make sure stream 2 is free  */
  cudaStreamSynchronize(cu_Stream[2]);

  /*  Copy results from device to host memory  */
  printf("Copying output correlation spectrum to host memory ...\n");
  cudaMemcpyAsync((void *)h_XOutput, (void *)d_XOutput, NBytes_DtoH, cudaMemcpyDeviceToHost, cu_Stream[2]);

  /***  END Copy output correlation matrix to host memory  ***/



  /***  BEGIN Write output to file  ***/

  /*  Take time stamp  */
  now_abs = time(0);
  now = localtime(&now_abs);


  /***  TEST: filtered and unfiltered spectra  ***/

  /*  Construct output file name  */
  strftime(outfilename, OUTFILENAMELEN*sizeof(char), "sim_data_%Y%m%d_%H%M%S_preCorr.csv", now);

  if ((outfile = fopen(outfilename, "w")) == NULL)
  {
    printf("Cannot open or create file %s!\n", outfilename);
    exit(10);
  }

  printf("Writing unfiltered and filtered spectra to file %s ...\n", outfilename);
  for (j=0 ; j<Nelements ; j++)
    for (p=0 ; p<Npols ; p++)
    {
      stride = (j*Npols + p) * Nchannels;

      for (i=0 ; i<Nchannels ; i++)
            fprintf(outfile, "%ld,%d,%ld,%.6f,%.6f,%.6f,%.6f\n", j, p, i, h_Unfiltered[stride + i].x, h_Unfiltered[stride + i].y, h_Filtered[stride + i].x, h_Filtered[stride + i].y);
            //  elementIdx, polIdx, channelIdx, Re_Unfiltered, Im_Unfiltered, Re_Filtered, Im_Filtered
    }

  /*  Close file  */
  fclose(outfile);


  /***  Output correlation matrix  ***/

  /*  Construct output file name  */
  strftime(outfilename, OUTFILENAMELEN*sizeof(char), "sim_data_%Y%m%d_%H%M%S.csv", now);

  if ((outfile = fopen(outfilename, "w")) == NULL)
  {
    printf("Cannot open or create file %s!\n", outfilename);
    exit(11);
  }

  /*  Sync with device so that asynchronous copy to host completes  */
  cudaStreamSynchronize(cu_Stream[2]);
  if ( (cu_RetVal = cudaErrCheck()) != 0 ) exit(cu_RetVal);

  printf("Writing output correlation matrix to file %s ...\n", outfilename);
  for (j=0 ; j<Nelements ; j++)  // element row index
    for (p=0 ; p<Npols ; p++)  // polarisation of element j
      for (k=j ; k<Nelements ; k++)  // element column index
        for (q=p ; q<Npols ; q++)  // polarisation of element k
          for (i=0 ; i<Nchannels ; i++)
          {
            stride_cublas = i*Npols*Nelements*Npols*Nelements + (k*Npols + q)*Npols*Nelements + (j*Npols + p);  // column-major array

            fprintf(outfile, "%ld,%d,%ld,%d,%ld,%.6f,%.6f\n", j, p, k, q, i, h_XOutput[stride_cublas].x, h_XOutput[stride_cublas].y);
            //  elementIdx_k, polIdx_q, elementIdx_j, polIdx_p, channelIdx, Re_GPU, Im_GPU
          }

  /*  Close file  */
  fclose(outfile);

  /***  END Write output to file  ***/


  printf("\n");
  return(0);
}
