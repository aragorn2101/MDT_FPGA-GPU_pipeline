#include "MDT_OneStation_client.h"


int main (int argc, char **argv)
{
  /***  BEGIN Variables for telescope  ***/

  int Nelements, Npols, Nchannels, Ntaps, Nspectra;
  int SperElement;  // number of complex samples per element per packet
  unsigned int NumPerPkt;  // number of 32-bit numbers in each packet
  unsigned long int sampleIdx;  // sample index in the grand scheme

  /*  Number of analogue inputs for each SNAP board in 1 station  */
  /*  Elements of array are in snapIdx order                      */
  unsigned short int InputsPerSNAP[4] = INPUTSPERSNAP;

  /***  END Variables for telescope  ***/



  /***  BEGIN Variables for array lengths in memory  ***/

  /*  Size of window for PFB  */
  long NBytes_Window;

  /*  Size of stacks in RAM  */
  long NBytes_Stacks;

  /*  Size of data copied from host to GPU at input  */
  long NBytes_HtoD;

  /*  Size of input array to F-engine  */
  long NBytes_FInput;

  /*  Size of data to shift in device memory between successive PFB's  */
  long NBytes_DtoD;

  /*  Total number of samples in FInput array  */
  long len_FInput;

  /*  Size of output array from F-engine  */
  long NBytes_FOutput;

  /*  Size of input array to the GPU X-engine  */
  long NBytes_XInput;

  /*  Size of cuBLAS pointer arrays  */
  long NBytes_cuBLASptr_Xin;
  long NBytes_cuBLASptr_Xout;

  /*  Size of output array copied from GPU to host at output  */
  long NBytes_DtoH;

  /*  Number of samples comprised in DtoH  */
  long len_XOutput;

  /***  END Variables for array lengths in memory  ***/



  /***  BEGIN Host variables  ***/

  /*  Stack for incoming signal data  */
  cuComplex *h_Stacks;

  /*  Pointer arrays for use with cuBLAS X-engine  */
  cuComplex **h_XInputPtr, **h_XOutputPtr;

  /*  cuBLAS scalar coefficients  */
  cuComplex cublas_alpha, cublas_beta;

  /*  Output array for GPU FX correlator (pinned)  */
  cuComplex *h_XOutput;


  long int i, j, k, z;
  unsigned short int p, q;
  unsigned long int Nsamples[2];  // samples written to array per element
  unsigned long int stride, stride_sample;
  unsigned int stride_buf;
  unsigned long int Samples_Nspectra;  // number of samples in Nspectra
  unsigned long int Samples_begins;  // (Ntaps - 1) spectra
  unsigned long int stride_Nspectra, stride_begins;
  unsigned long int stride_cublas;
  unsigned short int stackIdx, curr_stack;
  unsigned short int begins = 1;

  /*  Variables for output file  */
  FILE *outfile;
  char outfilename[OUTFILENAMELEN];
  struct tm *now;
  time_t now_abs;

  /***  END Host variables  ***/



  /***  BEGIN Device array variables  ***/

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

  /*  Variables for device properties  */
  int cu_devIdx;
  cudaDeviceProp cu_devProp;
  int cu_RetVal;
  char *cu_DeviceName;
  int cufft_version;
  int cu_MaxThreadsPerBlk;

  /*  GPU thread gridding parameters for device function PpS_Batch()  */
  int PpS_Batch_thdx, PpS_Batch_thdy, PpS_Batch_thdz;  // block dim
  int PpS_Batch_blkx, PpS_Batch_blky, PpS_Batch_blkz;  // grid dim

  /*  GPU thread gridding parameters for device function ReorderFOutput()  */
  int ReorderFOutput_thdx, ReorderFOutput_thdy, ReorderFOutput_thdz;  // block dim
  int ReorderFOutput_blkx, ReorderFOutput_blky, ReorderFOutput_blkz;  // block dim


  /*  CUDA Streams  */
  cudaStream_t cu_Stream;

  /*  cuFFT plan handle  */
  cufftHandle cu_FFT_Plan;

  /*  cuBLAS control  */
  cublasHandle_t cu_BLAS_XEngine;
  cublasStatus_t cu_BLAS_Stat;

  /***  END CUDA control variables  ***/



  /***  BEGIN MPI variables  ***/

  int mpi_NumThreads, mpi_threadIdx;
  char hostname[MPI_MAX_PROCESSOR_NAME];
  int mpi_len;
  int mpi_version, mpi_subversion;
  int mpi_tag;
  MPI_Status mpi_stat[4];
  MPI_Request mpi_req[4];
  int mpi_retval;
  int green_light = 1;

  /***  END MPI variables  ***/



  /***  BEGIN Variables for networking  ***/

  int listen_sock;
  struct addrinfo hints, *servinfo, *addrinfo_ptr;
  int retval;

  int numbytes;
  struct sockaddr_storage remote_addr;
  socklen_t addr_len = sizeof(remote_addr);
  char str_addr[INET6_ADDRSTRLEN];

  /*  Buffers to store packets from network socket  */
  unsigned int *udp_buf, *mpi_buf;

  /***  END Variables for networking  ***/



  /*
   *  Telescope mode: Standby
   *  Preliminary settings and system initialisation
   *
   */

  /*  Initialise MPI threads  */
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_threadIdx);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_NumThreads);


  /*  Retrieve device properties  */
  cu_RetVal = cudaGetDevice(&cu_devIdx);
  if ( (cu_RetVal = cudaErrCheck()) != 0 ) exit(cu_RetVal);

  cu_RetVal = cudaGetDeviceProperties(&cu_devProp, cu_devIdx);
  if ( (cu_RetVal = cudaErrCheck()) != 0 ) exit(cu_RetVal);

  cu_DeviceName = cu_devProp.name;  // Name of device
  cu_MaxThreadsPerBlk = cu_devProp.maxThreadsPerBlock;  // Maximum number of threads per block



  /***  BEGIN Master thread parses command line arguments  ***/

  if (mpi_threadIdx == MPIMASTERIDX)
  {
    /*  Check if required number of MPI threads is met  */
    if (mpi_NumThreads != REQNUMTHREADS)
    {
      printf("This program must run with %d threads!\n", REQNUMTHREADS);
      MPI_Abort(MPI_COMM_WORLD, 1);
      exit(1);
    }

    /*  Check number of command line arguments  */
    if (argc <= 6) // argc <= (number of expected arguments)
    {
      print_help(argv[0]);
      MPI_Abort(MPI_COMM_WORLD, 2);
      exit(2);
    }

    /*  Validate values input to program  */
    Nelements = atoi(argv[1]);
    if (Nelements < MINELEMENTS || Nelements > MAXELEMENTS)
    {
      printf("%s: Invalid number of elements!\n", argv[0]);
      print_help(argv[0]);
      MPI_Abort(MPI_COMM_WORLD, 3);
      exit(3);
    }

    /*  Number of polarisations for each element  */
    Npols = atoi(argv[2]);
    if (Npols < 1 || Npols > 2)
    {
      printf("%s: Invalid number of polarisations!\n", argv[0]);
      print_help(argv[0]);
      MPI_Abort(MPI_COMM_WORLD, 4);
      exit(4);
    }

    /*  Number of frequency channels in each spectrum  */
    Nchannels = atoi(argv[3]);
    if (Nchannels < MINCHANNELS || Nchannels > MAXCHANNELS)
    {
      printf("%s: Invalid number of frequency channels!\n", argv[0]);
      print_help(argv[0]);
      MPI_Abort(MPI_COMM_WORLD, 5);
      exit(5);
    }

    /*  Number of taps in PFB  */
    Ntaps = atoi(argv[4]);
    if (Ntaps < MINTAPS || Ntaps > MAXTAPS)
    {
      printf("%s: Invalid number of taps for PFB!\n", argv[0]);
      print_help(argv[0]);
      MPI_Abort(MPI_COMM_WORLD, 6);
      exit(6);
    }

    /*  Number of spectra for 1 integration time  */
    Nspectra = atoi(argv[5]);
    if (Nspectra < MINSPECTRA || Nspectra > MAXSPECTRA)
    {
      printf("%s: Invalid number of spectra!\n", argv[0]);
      print_help(argv[0]);
      MPI_Abort(MPI_COMM_WORLD, 7);
      exit(7);
    }

    /*  Number of complex samples per element in 1 packet  */
    SperElement = atoi(argv[6]);
    if (SperElement <= 0)
    {
      printf("%s: Invalid SperElement!\n", argv[0]);
      print_help(argv[0]);
      MPI_Abort(MPI_COMM_WORLD, 8);
      exit(8);
    }
  }

  /***  END Master thread parses command line arguments  ***/


  /***  SYNC MPI Threads  **/
  MPI_Barrier(MPI_COMM_WORLD);


  /***  Create CUDA stream for all MPI threads  ***/
  if (mpi_threadIdx != MPIMASTERIDX)
    cudaStreamCreate(&cu_Stream);


  /***  BEGIN Pipeline characteristics according to input arguments  ***/

  /*  All MPI threads take command line parameters  */
  Nelements = atoi(argv[1]);
  Npols = atoi(argv[2]);
  Nchannels = atoi(argv[3]);
  Ntaps = atoi(argv[4]);
  Nspectra = atoi(argv[5]);
  SperElement = atoi(argv[6]);


  /*  Number of 32-bit numbers in 1 UDP packet
   *  Includes the header
   *  This is an upper limit, assuming that the maximum number
   *  of inputs is at the first SNAP board
   */
  NumPerPkt = (2 + (InputsPerSNAP[0] * SperElement)) * 2;

  /*  Number of complex samples per element in 1 stack  */
  Samples_Nspectra = Nspectra * Nchannels;

  /*  Number of samples, per element, to accumulate in the beginning of
   *  acquisition due to the required first (Ntaps - 1) x Nchannels samples
   */
  Samples_begins = (Ntaps - 1) * Nchannels;

  /*  Strides due to the above  */
  stride_Nspectra = Samples_Nspectra * Npols*Nelements;
  stride_begins = Samples_begins * Npols*Nelements;
  len_FInput = stride_begins + stride_Nspectra;



  /***  BEGIN Host array sizes  ***/

  /*  Size of window for PFB  */
  NBytes_Window = sizeof(float) * Ntaps * Nchannels;

  /*  Size of stack  */
  NBytes_Stacks = sizeof(cuComplex) * 2 * Nspectra * Npols*Nelements * Nchannels;

  /*  Size of data copied from host to device at input  */
  NBytes_HtoD = sizeof(cuComplex) * Nspectra * Npols*Nelements * Nchannels;
  // input

  /*  Size of output array copied from device to host at output  */
  NBytes_DtoH = sizeof(cuComplex) * Npols*Nelements * Npols*Nelements * Nchannels;
  // output

  /*  Number of samples comprised in DtoH  */
  len_XOutput = Npols*Nelements * Npols*Nelements * Nchannels;

  /***  END Host array sizes  ***/


  /***  BEGIN Device array sizes  ***/

  /*  Size of input array to F-engine  */
  NBytes_FInput = sizeof(cuComplex) * (Nspectra + Ntaps - 1) * Npols*Nelements * Nchannels;

  /*  Size of data to shift in device memory between successive PFB's  */
  NBytes_DtoD = sizeof(cuComplex) * (Ntaps - 1) * Npols*Nelements * Nchannels;

  /*  Size of output array from F-engine  */
  NBytes_FOutput = sizeof(cufftComplex) * Nspectra * Npols*Nelements * Nchannels;

  /*  Size of input array to the GPU X-engine  */
  NBytes_XInput = sizeof(cuComplex) * Nspectra * Npols*Nelements * Nchannels;

  /*  Size of cuBLAS pointer arrays  */
  NBytes_cuBLASptr_Xin  = sizeof(cuComplex *) * Nchannels;
  NBytes_cuBLASptr_Xout = sizeof(cuComplex *) * Nchannels;

  /***  END Device array sizes  ***/

  /***  END Pipeline characteristics according to input arguments  ***/



  /***  BEGIN Master MPI thread verifies if packet size is within system limit and print info  ***/

  if (mpi_threadIdx == MPIMASTERIDX)
  {
    /*  Ensure that requested packet size according to SperElement does not
     *  exceed the 8.5 KiB limit for the network jumbo frame.
     */
    if (NumPerPkt*4 > 8704)
    {
      printf("%s: Requested Nelements and SperElement cause packet size to exceed 8.5KiB limit!\n", argv[0]);
      print_help(argv[0]);
      MPI_Abort(MPI_COMM_WORLD, 9);
      exit(9);
    }


    /*  If everything OK, print info and carry on  */
    MPI_Get_processor_name(hostname, &mpi_len);
    MPI_Get_version(&mpi_version, &mpi_subversion);
    cufftGetVersion(&cufft_version);

    printf("\n");
    printf("**************************************************\n");
    printf("                  Mode: Stand-by                  \n");
    printf("**************************************************\n\n");
    printf("Thread %d : Current machine: %s\n", mpi_threadIdx, hostname);
    printf("Thread %d : Total number of MPI threads: %d\n", mpi_threadIdx, mpi_NumThreads);
    printf("Thread %d : Version of MPI: %d.%d\n", mpi_threadIdx, mpi_version, mpi_subversion);
    printf("Thread %d : Using %s (ID: %d) as device.\n", mpi_threadIdx, cu_DeviceName, cu_devIdx);
    printf("Thread %d : GPU F-engine is powered by cuFFT version %d.\n", mpi_threadIdx, cufft_version);
    printf("\n");

  }

  /***  END Master MPI thread verifies if packet size is within system limit and print info  ***/


  /***  SYNC MPI Threads  **/
  MPI_Barrier(MPI_COMM_WORLD);


  switch(mpi_threadIdx)
  {
    /***  BEGIN Thread 0  ***/
    case 0:
    {
      /*  Create buffers to pass packets around  */
      udp_buf = (unsigned int *)malloc(NumPerPkt * sizeof(unsigned int));
      mpi_buf = (unsigned int *)malloc(NumPerPkt * sizeof(unsigned int));


      /***  BEGIN Setting up UDP socket for listen  ***/

      printf("Thread %d: Setting up socket ...\n", mpi_threadIdx);
      memset(&hints, 0, sizeof hints);
      hints.ai_family = AF_UNSPEC; // force IPv4
      hints.ai_socktype = SOCK_DGRAM;
      hints.ai_flags = AI_PASSIVE; // use my IP

      if ((retval = getaddrinfo(NULL, MYPORT, &hints, &servinfo)) != 0)
      {
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(retval));
        MPI_Abort(MPI_COMM_WORLD, 10);
        exit(10);
      }

      // loop through all the results and bind to the first we can
      for(addrinfo_ptr=servinfo; addrinfo_ptr!=NULL; addrinfo_ptr=addrinfo_ptr->ai_next)
      {
        if ((listen_sock = socket(addrinfo_ptr->ai_family, addrinfo_ptr->ai_socktype, addrinfo_ptr->ai_protocol)) == -1)
        {
          perror("socket");
          continue;
        }

        if (bind(listen_sock, addrinfo_ptr->ai_addr, addrinfo_ptr->ai_addrlen) == -1)
        {
          close(listen_sock);
          perror("bind");
          continue;
        }

        break;
      }

      if (addrinfo_ptr == NULL)
      {
        fprintf(stderr, "listener: failed to bind socket\n");
        MPI_Abort(MPI_COMM_WORLD, 11);
        exit(11);
      }
      /***  END Setting up UDP socket for listen  ***/


    }
    break;
    /***  END Thread 0  ***/


    /***  BEGIN Thread 1  ***/
    case 1:
    {
      /*  Create buffers to pass packets around  */
      udp_buf = (unsigned int *)malloc(NumPerPkt * sizeof(unsigned int));
      mpi_buf = (unsigned int *)malloc(NumPerPkt * sizeof(unsigned int));


      /***  BEGIN Host: Allocate arrays  ***/

      /*  Stack for incoming signal data (page-locked)  */
      cudaHostAlloc((void **)&h_Stacks, NBytes_Stacks, cudaHostAllocDefault);

      /***  END Host: Allocate arrays  ***/


      /***  BEGIN Device: Allocate arrays  ***/

      /*  Input array to F-engine  */
      cudaMalloc((void **)&d_FInput, NBytes_FInput);

      /***  END Device: Allocate arrays  ***/


      printf("Thread %d: Done with allocating arrays.\n", mpi_threadIdx);
    }
    break;
    /***  END Thread 1  ***/


    /***  BEGIN Thread 2  ***/
    case 2:
    {
      /***  BEGIN Host: Allocate arrays  ***/

      /*  Pointer arrays for use with cuBLAS X-engine  */
      h_XInputPtr  = (cuComplex **)malloc(NBytes_cuBLASptr_Xin);
      h_XOutputPtr = (cuComplex **)malloc(NBytes_cuBLASptr_Xout);

      /***  END Host: Allocate arrays  ***/


      /***  BEGIN Device: Allocate arrays  ***/

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

      /***  END Device: Allocate arrays  ***/


      printf("Thread %d: Done with allocating arrays.\n", mpi_threadIdx);


      /***  BEGIN Device: Initialise window coefficients for PFB  ***/

      printf("\nThread %d: Initialising window coefficients for PFB ...\n", mpi_threadIdx);
      printf("Thread %d: Grid configuration  = (%6d,%6d,%6d)\n", mpi_threadIdx, cu_MaxThreadsPerBlk, 1, 1);
      printf("Thread %d: Block configuration = (%6d,%6d,%6d)\n", mpi_threadIdx, (Ntaps*Nchannels)/cu_MaxThreadsPerBlk + (((Ntaps*Nchannels)%cu_MaxThreadsPerBlk != 0) ? 1 : 0), 1, 1);
      PFB_Window_ExBlackman<<< cu_MaxThreadsPerBlk, (Ntaps*Nchannels)/cu_MaxThreadsPerBlk + (((Ntaps*Nchannels)%cu_MaxThreadsPerBlk != 0) ? 1 : 0), 0, cu_Stream >>>(Nchannels, Ntaps, d_Window);
      cudaStreamSynchronize(cu_Stream);
      if ((cu_RetVal = cudaErrCheck()) != 0)
      {
        MPI_Abort(MPI_COMM_WORLD, 12);
        exit(12);
      }
      printf("Thread %d: Done.\n", mpi_threadIdx);

      /***  END Device: Initialise window coefficients for PFB  ***/



      /***  BEGIN Device: PpS_Batch() grid characteristics  ***/

      printf("\nThread %d: Computing GPU grid structure for  the PpS_Batch() kernel ...\n", mpi_threadIdx);
      PpS_Batch_thdx = cu_MaxThreadsPerBlk;
      PpS_Batch_thdy = 1;
      PpS_Batch_thdz = 1;

      PpS_Batch_blkx = Nchannels/PpS_Batch_thdx + ((Nchannels%PpS_Batch_thdx != 0) ? 1 : 0);
      PpS_Batch_blky = Nspectra;
      PpS_Batch_blkz = Npols*Nelements;
      printf("Thread %d: Grid configuration  = (%6d,%6d,%6d)\n", mpi_threadIdx, PpS_Batch_blkx, PpS_Batch_blky, PpS_Batch_blkz);
      printf("Thread %d: Block configuration = (%6d,%6d,%6d)\n", mpi_threadIdx, PpS_Batch_thdx, PpS_Batch_thdy, PpS_Batch_thdz);

      /***  END Device: PpS_Batch() grid characteristics  ***/



      /***  BEGIN Device: prepare cuFFT plan  ***/

      printf("\nThread %d: Constructing cuFFT plan prior to processing ...\n", mpi_threadIdx);
      if ((cu_RetVal = cufftPlan1d(&cu_FFT_Plan, Nchannels, CUFFT_C2C, Nspectra*Npols*Nelements)) != CUFFT_SUCCESS)
      {
        fprintf(stderr, "CUFFT error (%d): failed to create FFT plan for F-engine.\n", cu_RetVal);
        MPI_Abort(MPI_COMM_WORLD, 13);
        exit(13);
      }

      printf("\nThread %d: Setting up stream for cuFFT ...\n", mpi_threadIdx);
      cufftSetStream(cu_FFT_Plan, cu_Stream);

      /***  END Device: prepare cuFFT plan  ***/



      /***  BEGIN Device: ReorderFOutput() grid characteristics  ***/

      printf("\nThread %d: Computing GPU grid structure for  the ReorderFOutput() kernel ...\n", mpi_threadIdx);
      ReorderFOutput_thdx = 32;
      ReorderFOutput_thdy = 32;
      ReorderFOutput_thdz = 1;

      ReorderFOutput_blkx = Nspectra;
      ReorderFOutput_blky = Nchannels/ReorderFOutput_thdy + ((Nchannels%ReorderFOutput_thdy != 0) ? 1 : 0);
      ReorderFOutput_blkz = (Npols*Nelements)/ReorderFOutput_thdx + (((Npols*Nelements)%ReorderFOutput_thdx != 0) ? 1 : 0);
      printf("Thread %d: Grid configuration  = (%6d,%6d,%6d)\n", mpi_threadIdx, ReorderFOutput_blkx, ReorderFOutput_blky, ReorderFOutput_blkz);
      printf("Thread %d: Block configuration = (%6d,%6d,%6d)\n", mpi_threadIdx, ReorderFOutput_thdx, ReorderFOutput_thdy, ReorderFOutput_thdz);

      /***  END Device: ReorderFOutput() grid characteristics  ***/



      /***  BEGIN Device: Initialise cuBLAS  ***/

      printf("\nThread %d: Initialising pointer arrays for use with cuBLAS ...\n", mpi_threadIdx);
      for (i=0 ; i<Nchannels ; i++)
      {
        h_XInputPtr[i]  = d_XInput  + i*Npols*Nelements*Nspectra;
        h_XOutputPtr[i] = d_XOutput + i*Npols*Nelements*Npols*Nelements;
      }

      /*  Copy pointer arrays to device  */
      printf("Copying cuBLAS pointer arrays to device ...\n");
      cudaMemcpy((void **)d_XInputPtr, (void **)h_XInputPtr, NBytes_cuBLASptr_Xin, cudaMemcpyHostToDevice);
      cudaDeviceSynchronize();
      if ((cu_RetVal = cudaErrCheck()) != 0)
      {
        MPI_Abort(MPI_COMM_WORLD, 14);
        exit(14);
      }
      cudaMemcpy((void **)d_XOutputPtr, (void **)h_XOutputPtr, NBytes_cuBLASptr_Xout, cudaMemcpyHostToDevice);
      cudaDeviceSynchronize();
      if ((cu_RetVal = cudaErrCheck()) != 0)
      {
        MPI_Abort(MPI_COMM_WORLD, 15);
        exit(15);
      }

      free(h_XInputPtr);
      free(h_XOutputPtr);

      /*  Initialise scalar coefficients  */
      cublas_alpha.x = 1.0f / Nspectra;
      cublas_alpha.y = 0.0f;
      cublas_beta.x  = 0.0f;
      cublas_beta.y  = 0.0f;

      /*  Initialise cuBLAS and set stream  */
      printf("\nThread %d: Initialising cuBLAS X-engine and associating stream ...\n", mpi_threadIdx);
      cu_BLAS_Stat = cublasCreate(&cu_BLAS_XEngine);
      cudaDeviceSynchronize();
      if (cu_BLAS_Stat != CUBLAS_STATUS_SUCCESS)
      {
        printf("\nThread %d: Failed to initialise cuBLAS (cublasStat = %d)!\n", mpi_threadIdx, (int)cu_BLAS_Stat);
        MPI_Abort(MPI_COMM_WORLD, 16);
        exit(16);
      }

      cu_BLAS_Stat = cublasSetStream(cu_BLAS_XEngine, cu_Stream);
      if (cu_BLAS_Stat != CUBLAS_STATUS_SUCCESS)
      {
        printf("\nThread %d: Failed to set cuBLAS stream (cublasStat = %d)!\n", mpi_threadIdx, (int)cu_BLAS_Stat);
        MPI_Abort(MPI_COMM_WORLD, 17);
        exit(17);
      }

      /***  END Device: Initialise cuBLAS  ***/


      printf("Thread %d: Done.\n", mpi_threadIdx);

    }
    break;
    /***  END Thread 2  ***/


    /***  BEGIN Thread 3  ***/
    case 3:
    {
      /***  BEGIN Host: Allocate arrays  ***/

      cudaHostAlloc((void **)&h_XOutput, NBytes_DtoH, cudaHostAllocDefault);

      /***  END Host: Allocate arrays  ***/


      /***  BEGIN Device: Allocate arrays  ***/

      /*  Input array to F-engine  */
      cudaMalloc((void **)&d_FInput, NBytes_FInput);

      /*  Output array for X-engine  */
      cudaMalloc((void **)&d_XOutput, NBytes_DtoH);

      /***  END Device: Allocate arrays  ***/


      printf("Thread %d: Done with allocating arrays.\n", mpi_threadIdx);
    }
    break;
    /***  END Thread 3  ***/

  }


  /***  SYNC MPI Threads  **/
  MPI_Barrier(MPI_COMM_WORLD);
  if (mpi_threadIdx == MPIMASTERIDX)
      printf("\nThread %d: Finished system initialisation.\n\n", mpi_threadIdx);



  /*
   *  Telescope mode: Observing
   *
   */

  switch(mpi_threadIdx)
  {
    /***  BEGIN Thread 0  ***/
    case 0:
    {
      printf("\n");
      printf("**************************************************\n");
      printf("                  Mode: Observing                 \n");
      printf("**************************************************\n\n");

      printf("Thread %d: Listening for connections on UDP socket ...\n", mpi_threadIdx);

      /***  First Listen and transfer to thread 1  ***/
      if ((numbytes = recvfrom(listen_sock, udp_buf, NumPerPkt * sizeof(unsigned int), 0, (struct sockaddr *)&remote_addr, &addr_len)) == -1)
      {
        perror("recvfrom");
        MPI_Abort(MPI_COMM_WORLD, 18);
        exit(18);
      }


      /*  Transfer packet to MPI buffer  */
      memcpy((void *)mpi_buf, (const void *)udp_buf, NumPerPkt * sizeof(unsigned int));

      /*  Send packet to MPI thread 1  */
      mpi_tag = 0;
      MPI_Isend((const void *)mpi_buf, NumPerPkt, MPI_UINT32_T, 1, mpi_tag, MPI_COMM_WORLD, &mpi_req[0]);
             /* &buffer, count, datatype, dest, tag, comm, &request  */



      /***  BEGIN LOOP Listen and transfer packet  ***/
      while (1)
      {
        /*  Listen  */
        if ((numbytes = recvfrom(listen_sock, udp_buf, NumPerPkt * sizeof(unsigned int), 0, (struct sockaddr *)&remote_addr, &addr_len)) == -1)
        {
          perror("recvfrom");
          MPI_Abort(MPI_COMM_WORLD, 19);
          exit(19);
        }

        /*  Block if the previous Isend did not finish  */
        MPI_Wait(&mpi_req[0], &mpi_stat[0]);

        /*  Transfer packet to MPI buffer  */
        memcpy((void *)mpi_buf, (const void *)udp_buf, NumPerPkt * sizeof(unsigned int));

        /*  Send packet to MPI thread 1  */
        // mpi_tag = 0
        MPI_Isend((const void *)mpi_buf, NumPerPkt, MPI_UINT32_T, 1, mpi_tag, MPI_COMM_WORLD, &mpi_req[0]);
               /* &buffer, count, datatype, dest, tag, comm, &request  */

      }
      /***  END LOOP Listen and transfer packet  ***/
    }
    break;
    /***  END Thread 0  ***/


    /***  BEGIN Thread 1  ***/
    case 1:
    {
      Nsamples[0] = 0;
      Nsamples[1] = 0;

      /***  BEGIN First (Ntaps - 1) x Nchannels samples  ***/

      /* First receive packet from MPI thread 0  */
      mpi_tag = 0;
      MPI_Irecv((void *)mpi_buf, NumPerPkt, MPI_UINT32_T, 0, mpi_tag, MPI_COMM_WORLD, &mpi_req[0]);
                   /*  &buf, count, dtype, src, tag, comm, &stat  */

      do
      {
        /*  Wait for non-blocking recv to conclude  */
        MPI_Wait(&mpi_req[0], &mpi_stat[0]);

        /*  Transfer packet to MPI buffer  */
        memcpy((void *)udp_buf, (const void *)mpi_buf, NumPerPkt * sizeof(unsigned int));

        /*  Non-blocking Recv to wait for thread 0 to pass packet  */
        // mpi_tag = 0
        MPI_Irecv((void *)mpi_buf, NumPerPkt, MPI_UINT32_T, 0, mpi_tag, MPI_COMM_WORLD, &mpi_req[0]);
                     /*  &buf, count, dtype, src, tag, comm, &request  */



        /***  BEGIN Depacketise  ***/

        //  Data array is using the following layout
        //  Stack -> Spectrum -> Element -> Pol -> Channel

        for (i=0 ; i<SperElement ; i++)
        {
          /*  Calculate index for the current samples  */
          sampleIdx = (udp_buf[2] * SperElement) + i;  // element sample index in the grand scheme
          // spectrumIdx = sampleIdx / Nchannels
          // channelIdx  =  sampleIdx % Nchannels
          // stackIdx = (spectrumIdx / Nspectra) % 2

          /*  Cater for first (Ntaps - 1) x Nchannels samples at the beginning  */
          if (sampleIdx < Samples_begins)
            stackIdx = 1;
          else
          {
            sampleIdx -= Samples_begins;
            stackIdx = 0;
          }


          /*  Calculate the stride where to write data in array  */
          //  Stride due to stack:
          //    stackIdx * Nspectra * Npols*Nelements * Nchannels
          //  Spectrum stride:
          //    (sampleIdx / Nchannels) * Npols*Nelements * Nchannels
          stride = (stackIdx*Nspectra + (sampleIdx / Nchannels)) * Npols*Nelements * Nchannels;

          //  Element stride due to stationIdx
          stride += (udp_buf[0] * INPUTSPERSTATION) * Nchannels;

          //  Element stride due to snapIdx
          for (k=0 ; k<udp_buf[1] ; k++)
            stride += (InputsPerSNAP[k] * Nchannels);

          //  Channel stride
          stride += (sampleIdx % Nchannels);


          /*  Loop over elements in this packet  */
          for (j=0 ; j<InputsPerSNAP[udp_buf[1]] ; j++)
          {
            /*  Stride for current sample in MPI buffer  */
            stride_buf = 4 + 2*(i*InputsPerSNAP[udp_buf[1]] + j);

            /*  Stride for current sample in data array  */
            stride_sample = stride + j*Nchannels;

            // Convert back to signed integer, then float and shift point by 19 dec.
            // Re
            if (udp_buf[stride_buf] >= 2147483648)
              h_Stacks[stride_sample].x = -1.9073486328125e-06 * (int)(4294967295 - udp_buf[stride_buf] + 1);
            else
              h_Stacks[stride_sample].x =  1.9073486328125e-06 * (int)(udp_buf[stride_buf]);

            // Im
            if (udp_buf[stride_buf + 1] >= 2147483648)
              h_Stacks[stride_sample].y = -1.9073486328125e-06 * (int)(4294967295 - udp_buf[stride_buf + 1] + 1);
            else
              h_Stacks[stride_sample].y =  1.9073486328125e-06 * (int)(udp_buf[stride_buf + 1]);
          }

          Nsamples[stackIdx]++;
        }
      }
      while (Nsamples[1] < Samples_begins);

      /*  Transfer the first (Ntaps - 1) x Nchannels samples to data array on device  */
      cudaMemcpyAsync((void *)d_FInput, (void *)&h_Stacks[stride_Nspectra], NBytes_DtoD, cudaMemcpyHostToDevice, cu_Stream);

      /*  Update stack status  */
      Nsamples[1] = 0;

      /***  END Depacketise  ***/

      /***  END First (Ntaps - 1) x Nchannels samples  ***/



      /***  BEGIN LOOP Service subsequent packets from MPI thread 0  ***/

      curr_stack = 0;  // current data stack being prepared to send to GPU
      while (1)
      {
        /*  Wait for non-blocking recv to conclude  */
        MPI_Wait(&mpi_req[0], &mpi_stat[0]);

        /*  Transfer packet to MPI buffer  */
        memcpy((void *)udp_buf, (const void *)mpi_buf, NumPerPkt * sizeof(unsigned int));

        /*  Non-blocking Recv to wait for thread 0 to pass packet  */
        mpi_tag = 0;
        MPI_Irecv((void *)mpi_buf, NumPerPkt, MPI_UINT32_T, 0, mpi_tag, MPI_COMM_WORLD, &mpi_req[0]);
                     /*  &buf, count, dtype, src, tag, comm, &request  */



        /***  BEGIN Depacketise  ***/

        //  Data array is using the following layout
        //  Stack -> Spectrum -> Element -> Pol -> Channel

        for (i=0 ; i<SperElement ; i++)
        {
          /*  Calculate indices for the current sample  */
          sampleIdx = (udp_buf[2] * SperElement) - Samples_begins + i;  // element sample index in the grand scheme
          // spectrumIdx = sampleIdx / Nchannels
          // channelIdx  =  sampleIdx % Nchannels
          // stackIdx = (spectrumIdx / Nspectra) % 2

          /*  Calculate which part of the stack sample should be placed  */
          stackIdx   = (sampleIdx / Samples_Nspectra) % 2;  // either 0 or 1

          /*  Calculate the stride where to write data in array  */
          //  Stride due to stack:
          //    stackIdx * Nspectra * Npols*Nelements * Nchannels
          //  Spectrum stride:
          //    ((sampleIdx / Nchannels) % Nspectra) * Npols*Nelements * Nchannels
          stride = (stackIdx*Nspectra + ((sampleIdx / Nchannels) % Nspectra)) * Npols*Nelements * Nchannels;

          //  Element stride due to stationIdx
          stride += (udp_buf[0] * INPUTSPERSTATION) * Nchannels;

          //  Element stride due to snapIdx
          for (k=0 ; k<udp_buf[1] ; k++)
            stride += (InputsPerSNAP[k] * Nchannels);

          //  Channel stride
          stride += (sampleIdx % Nchannels);


          /*  Loop over elements in this packet  */
          for (j=0 ; j<InputsPerSNAP[udp_buf[1]] ; j++)
          {
            /*  Stride for current sample in MPI buffer  */
            stride_buf = 4 + 2*(i*InputsPerSNAP[udp_buf[1]] + j);

            /*  Stride for current sample in data array  */
            stride_sample = stride + j*Nchannels;

            // Convert back to signed integer, then float and shift point by 19 dec.
            // Re
            if (udp_buf[stride_buf] >= 2147483648)
              h_Stacks[stride_sample].x = -1.9073486328125e-06 * (int)(4294967295 - udp_buf[stride_buf] + 1);
            else
              h_Stacks[stride_sample].x =  1.9073486328125e-06 * (int)(udp_buf[stride_buf]);

            // Im
            if (udp_buf[stride_buf + 1] >= 2147483648)
              h_Stacks[stride_sample].y = -1.9073486328125e-06 * (int)(4294967295 - udp_buf[stride_buf + 1] + 1);
            else
              h_Stacks[stride_sample].y =  1.9073486328125e-06 * (int)(udp_buf[stride_buf + 1]);
          }

          Nsamples[stackIdx]++;
        }

        /***  END Depacketise  ***/


        /***  BEGIN Transfer data to device if current stack full  ***/
        if (Nsamples[curr_stack] == Samples_Nspectra)
        {
          if (begins == 1)
          {
            /*  Sync with device to see if last cudaMemcpyAsync finished  */
            cudaStreamSynchronize(cu_Stream);
            if ((cu_RetVal = cudaErrCheck()) != 0)
            {
              MPI_Abort(MPI_COMM_WORLD, 20);
              exit(20);
            }

            /*  Transfer content of current stack to data array  */
            cudaMemcpyAsync((void *)&d_FInput[stride_begins], (void *)&h_Stacks[curr_stack * stride_Nspectra], NBytes_HtoD, cudaMemcpyHostToDevice, cu_Stream);

            begins = 0;
          }
          else
          {
            /*  Sync with device to see if last cudaMemcpyAsync finished  */
            cudaStreamSynchronize(cu_Stream);
            if ((cu_RetVal = cudaErrCheck()) != 0)
            {
              MPI_Abort(MPI_COMM_WORLD, 21);
              exit(21);
            }

            /*  Share d_FInput with MPI thread 2 to trigger processing  */
            mpi_tag = 1;
            MPI_Isend((const void *)d_FInput, len_FInput, MPI_UNSIGNED_LONG, 2, mpi_tag, MPI_COMM_WORLD, &mpi_req[1]);
                   /* &buffer, count, datatype, dest, tag, comm, &request  */


            /*  Receive device pointers from MPI thread 3 to be able to carry on  */
            mpi_tag = 3;
            MPI_Recv((void *)d_FInput, len_FInput, MPI_UNSIGNED_LONG, 3, mpi_tag, MPI_COMM_WORLD, &mpi_stat[1]);
                         /*  &buf, count, dtype, src, tag, comm, &stat  */

            /*  If lights are green, transfer content of current stack to data array  */
            cudaMemcpyAsync((void *)&d_FInput[stride_begins], (void *)&h_Stacks[curr_stack * stride_Nspectra], NBytes_HtoD, cudaMemcpyHostToDevice, cu_Stream);

          }

          /*  Update stack status  */
          Nsamples[curr_stack] = 0;

          /*  Swap current stack being serviced for GPU  */
          curr_stack = (curr_stack == 0);
        }
        /***  END Transfer data to device if current stack full  ***/

      }
      /***  END LOOP Service subsequent packets from MPI thread 0  ***/

    }
    break;
    /***  END Thread 1  ***/


    /***  BEGIN Thread 2  ***/
    case 2:
    {
      while (1)
      {
        /*  Receive d_FInput pointers to begin processing  */
        mpi_tag = 1;
        MPI_Recv((void *)d_FInput, len_FInput, MPI_UNSIGNED_LONG, 1, mpi_tag, MPI_COMM_WORLD, &mpi_stat[0]);
                     /*  &buf, count, dtype, src, tag, comm, &stat  */



        /***  BEGIN Device: F-engine  ***/

        /***  Compute polyphase structure  ***/
        PpS_Batch<<< dim3(PpS_Batch_blkx,PpS_Batch_blky,PpS_Batch_blkz), dim3(PpS_Batch_thdx,PpS_Batch_thdy,PpS_Batch_thdz), 0, cu_Stream >>>(Nchannels, Ntaps, d_Window, d_FInput, d_FOutput);


        /*  Sync with device to see if PpS_Batch finished  */
        cudaStreamSynchronize(cu_Stream);
        if ((cu_RetVal = cudaErrCheck()) != 0)
        {
          MPI_Abort(MPI_COMM_WORLD, 22);
          exit(22);
        }


        /*  Share input array pointers with MPI thread 3 for (Ntaps - 1) shift  */
        if (begins != 1)
          MPI_Wait(&mpi_req[1], &mpi_stat[1]); // Wait for non-blocking send to conclude

        mpi_tag = 2;
        MPI_Isend((const void *)d_FInput, len_FInput, MPI_UNSIGNED_LONG, 3, mpi_tag, MPI_COMM_WORLD, &mpi_req[1]);
               /* &buffer, count, datatype, dest, tag, comm, &request  */


        /***  In-place Fast Fourier Transform  ***/
        cufftExecC2C(cu_FFT_Plan, d_FOutput, d_FOutput, CUFFT_FORWARD);

        /***  END Device: F-engine  ***/



        /***  BEGIN Device: X-engine  ***/

        /*  Re-order array for input to cuBLAS function  */
        ReorderFOutput<<< dim3(ReorderFOutput_blkx,ReorderFOutput_blky,ReorderFOutput_blkz), dim3(ReorderFOutput_thdx,ReorderFOutput_thdy,ReorderFOutput_thdz), 0, cu_Stream >>>(Nelements, Npols, Nchannels, d_FOutput, d_XInput);


        /*  Block until MPI thread 3 gives green light to overwrite device output array  */
        if (begins != 1)
        {
          mpi_tag = 5;
          MPI_Recv((void *)&green_light, 1, MPI_INT, 3, mpi_tag, MPI_COMM_WORLD, &mpi_stat[2]);
                       /*  &buf, count, dtype, src, tag, comm, &stat  */
        }


        /*  Cross-Correlation engine using cuBLAS  ***/
        cu_BLAS_Stat = CUBLAS_STATUS_EXECUTION_FAILED;
        cu_BLAS_Stat = cublasCgemmBatched(cu_BLAS_XEngine, CUBLAS_OP_N, CUBLAS_OP_C, Npols*Nelements, Npols*Nelements, Nspectra, &cublas_alpha, (const cuComplex **)d_XInputPtr, Npols*Nelements, (const cuComplex **)d_XInputPtr, Npols*Nelements, &cublas_beta, d_XOutputPtr, Npols*Nelements, Nchannels);
        cudaStreamSynchronize(cu_Stream);
        if (cu_BLAS_Stat != CUBLAS_STATUS_SUCCESS)
        {
          printf("\nThread %d: cuBLAS X-engine failed (cublasStat = %d)!\n", mpi_threadIdx, (int)cu_BLAS_Stat);
          MPI_Abort(MPI_COMM_WORLD, 23);
          exit(23);
        }

        /***  END Device: X-engine  ***/



        /*  Signal MPI thread 3 that processing is done and it can carry on
         *  with writing output to file.
         */
        if (begins != 1)
          MPI_Wait(&mpi_req[3], &mpi_stat[3]); // Wait for non-blocking send to conclude
        else
          begins = 0;

        mpi_tag = 4;
        MPI_Isend((const void *)d_XOutput, len_XOutput, MPI_UNSIGNED_LONG, 3, mpi_tag, MPI_COMM_WORLD, &mpi_req[3]);
               /* &buffer, count, datatype, dest, tag, comm, &request  */
      }
    }
    break;
    /***  END Thread 2  ***/


    /***  BEGIN Thread 3  ***/
    case 3:
    {
      /***  BEGIN LOOP Copy data/output and write to file  ***/
      while (1)
      {
        /*  Receive d_FInput pointers from MPI thread 2  */
        mpi_tag = 2;
        MPI_Recv((void *)d_FInput, len_FInput, MPI_UNSIGNED_LONG, 2, mpi_tag, MPI_COMM_WORLD, &mpi_stat[0]);
                     /*  &buf, count, dtype, src, tag, comm, &stat  */

        /*  Copy last (Ntaps - 1) spectra to the beginning of input array to PFB  */
        cudaMemcpyAsync((void *)d_FInput, (void *)&d_FInput[stride_Nspectra], NBytes_DtoD, cudaMemcpyDeviceToDevice, cu_Stream);


        /*  Sync with device to see if copying completed  */
        cudaStreamSynchronize(cu_Stream);
        if ((cu_RetVal = cudaErrCheck()) != 0)
        {
          MPI_Abort(MPI_COMM_WORLD, 24);
          exit(24);
        }

        /*  If cudaMemcpyDeviceToDevice finished, share d_FInput device
         *  pointers with MPI thread 1  */
        if (begins != 1)
          MPI_Wait(&mpi_req[1], &mpi_stat[1]); // Wait for non-blocking send to conclude

        mpi_tag = 3;
        MPI_Isend((const void *)d_FInput, len_FInput, MPI_UNSIGNED_LONG, 1, mpi_tag, MPI_COMM_WORLD, &mpi_req[1]);
               /* &buffer, count, datatype, dest, tag, comm, &request  */



        /*  Receive d_XOutput device pointers to retrieve results  */
        mpi_tag = 4;
        MPI_Recv((void *)d_XOutput, len_XOutput, MPI_UNSIGNED_LONG, 2, mpi_tag, MPI_COMM_WORLD, &mpi_stat[2]);
                     /*  &buf, count, dtype, src, tag, comm, &stat  */


        /*  Copy results from device to host memory  */
        cudaMemcpyAsync((void *)h_XOutput, (void *)d_XOutput, NBytes_DtoH, cudaMemcpyDeviceToHost, cu_Stream);


        /***  BEGIN Write output to file  ***/

        /*  Take time stamp  */
        now_abs = time(0);
        now = localtime(&now_abs);

        /*  Construct output file name  */
        strftime(outfilename, OUTFILENAMELEN*sizeof(char), "MDT_%Y%m%d_%H%M%S.csv", now);

        if ((outfile = fopen(outfilename, "w")) == NULL)
        {
          printf("Cannot open or create file %s!\n", outfilename);
          MPI_Abort(MPI_COMM_WORLD, 25);
          exit(25);
        }


        /*  Sync with device so that asynchronous copy to host completes  */
        cudaStreamSynchronize(cu_Stream);
        if ((cu_RetVal = cudaErrCheck()) != 0)
        {
          MPI_Abort(MPI_COMM_WORLD, 26);
          exit(26);
        }

        /*  Signal MPI thread 2 that it can overwrite the device output array  */
        if (begins != 1)
          MPI_Wait(&mpi_req[3], &mpi_stat[3]); // Wait for non-blocking send to conclude
        else
          begins = 0;

        mpi_tag = 5;
        MPI_Isend((const void *)&green_light, 1, MPI_INT, 2, mpi_tag, MPI_COMM_WORLD, &mpi_req[3]);
               /* &buffer, count, datatype, dest, tag, comm, &request  */


        printf("Writing output to file %s ...\n", outfilename);
        for (j=0 ; j<Nelements ; j++)  // element row index
          for (p=0 ; p<Npols ; p++)  // polarisation of element j
            for (k=j ; k<Nelements ; k++)  // element column index
              for (q=p ; q<Npols ; q++)  // polarisation of element k
              {

                /*  CUFFT will put the positive frequency spectrum at the first
                    Nchannels/2 positions in the array, then the negative
                    frequencies will follow. Therefore, the first B/2 frequencies
                    are at the end of the array  */
                z = 0;
                for (i=Nchannels/2 ; i<Nchannels ; i++)
                {
                  stride_cublas = i*Npols*Nelements*Npols*Nelements + (k*Npols + q)*Npols*Nelements + (j*Npols + p);  // column-major array

                  fprintf(outfile, "%ld,%d,%ld,%d,%ld,%.6f,%.6f\n", j, p, k, q, z, h_XOutput[stride_cublas].x, h_XOutput[stride_cublas].y);
                  //  elementIdx_k, polIdx_q, elementIdx_j, polIdx_p, channelIdx, Re_GPU, Im_GPU
                  z++;
                }

                for (i=0 ; i<Nchannels/2 ; i++)
                {
                  stride_cublas = i*Npols*Nelements*Npols*Nelements + (k*Npols + q)*Npols*Nelements + (j*Npols + p);  // column-major array

                  fprintf(outfile, "%ld,%d,%ld,%d,%ld,%.6f,%.6f\n", j, p, k, q, z, h_XOutput[stride_cublas].x, h_XOutput[stride_cublas].y);
                  //  elementIdx_k, polIdx_q, elementIdx_j, polIdx_p, channelIdx, Re_GPU, Im_GPU
                  z++;
                }
            }

        /*  Close file  */
        fclose(outfile);

        /***  END Write output to file  ***/
      }
      /***  END LOOP Copy data/output and write to file  ***/
    }
    break;
    /***  END Thread 3  ***/
  }

  /*  Close all MPI instances  */
  mpi_retval = MPI_Finalize();

  printf("\n");
  return(mpi_retval);
}
