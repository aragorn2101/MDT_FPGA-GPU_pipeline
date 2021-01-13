#ifndef MDT_HOST_FUNC
#define MDT_HOST_FUNC

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

/*  Limiting parameters for processing pipeline  */
#define MINELEMENTS 3  // minimum number of elements in array
#define MAXELEMENTS 2048  // maximum number of elements in array
#define MINCHANNELS 32  // minimum number of output frequency channels
#define MAXCHANNELS 16384  // maximum number of output frequency channels
#define MINTAPS 4  // minimum number of taps in polyphase structure
#define MAXTAPS 128  // maximum number of taps in polyphase structure
#define MINSPECTRA 2  // minimum number of spectra to compute
#define MAXSPECTRA 16384  // maximum number of spectra to compute


/*  Host function prototypes  */
void print_help(int, char *);
int cudaErrCheck();

#endif
