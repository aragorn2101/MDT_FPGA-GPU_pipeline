#include "MDT_host_functions.h"

/*
 *  Function to print a help message and exit program.
 *
 */
void print_help(char *PRGNAM)
{
  printf("\nUsage: mpirun --use-hwthread-cpus %s NELEMENTS NPOLS NCHANNELS NTAPS NSPECTRA SPERELEMENT\n", PRGNAM);
  printf("NELEMENTS: number of elements in the array from which signal is taken (%d <= NELEMENTS <= %d),\n", MINELEMENTS, MAXELEMENTS);
  printf("NPOLS: number of polarisation available for each element, NPOLS = 1 or 2,\n");
  printf("NCHANNELS: number of frequency channels in each output power spectrum (%d <= NCHANNELS <= %d),\n", MINCHANNELS, MAXCHANNELS);
  printf("NTAPS: number of taps for polyphase filter bank (%d <= NTAPS <= %d),\n", MINTAPS, MAXTAPS);
  printf("NSPECTRA: number of spectra over which average is calculated (%d <= NSPECTRA <= %d)\n", MINSPECTRA, MAXSPECTRA);
  printf("SPERELEMENT: number of complex samples per element in 1 UDP packet.\n");
  printf("Since the correlator assumes an FX design, the total number of time samples\n");
  printf("over which average will be done is (NCHANNELS x NSPECTRA).\n");
}



/*
 *  CPU function to check for CUDA errors
 *  in kernel return values.
 *
 */
int cudaErrCheck()
{
  /*  Get error code from previous function call  */
  cudaError_t retval = cudaPeekAtLastError();

  if (retval == cudaSuccess)
    return(0);
  else
  {
    printf("CUDA Error Code %d : %s\n", retval, cudaGetErrorName(retval));
    printf("  %s\n\n", cudaGetErrorString(retval));

    return((int)retval);
  }
}



// Get sockaddr, IPv4 or IPv6
void *get_address(struct sockaddr *sa)
{
  if (sa->sa_family == AF_INET6)
    return &( ( (struct sockaddr_in6*)sa )->sin6_addr );
  else
    return &( ( (struct sockaddr_in*)sa )->sin_addr );
}
