#ifndef MDT_ONESTATION_CLIENT
#define MDT_ONESTATION_CLIENT


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include "GaussianRNG.h"
#include "MDT_host_functions.h"
#include "MDT_device_functions.h"


/*  Limiting parameters for processing pipeline  */
#define MINELEMENTS 3  // minimum number of elements in array
#define MAXELEMENTS 2048  // maximum number of elements in array
#define MINCHANNELS 32  // minimum number of output frequency channels
#define MAXCHANNELS 16384  // maximum number of output frequency channels
#define MINTAPS 4  // minimum number of taps in polyphase structure
#define MAXTAPS 128  // maximum number of taps in polyphase structure
#define MINSPECTRA 2  // minimum number of spectra to compute
#define MAXSPECTRA 16384  // maximum number of spectra to compute

/*  Number of analogue inputs per station  */
#define INPUTSPERSTATION 38

/*  Number of analogue inputs for each SNAP board in 1 station  */
/*  Elements of array are in snapIdx order                      */
#define INPUTSPERSNAP {10, 10, 10, 8}

/*  Index for MPI master thread  */
#define MPIMASTERIDX 0

/*  Number of threads required for this program to run  */
#define REQNUMTHREADS 4

/*  Port to listen for UDP packets  */
#define MYPORT "52000"

/*  Length for output file name  */
#define OUTFILENAMELEN 80

/*  Maximum available RAM in bytes (B) on host  */
#define MAXHOSTRAM (14L * 1024L * 1024L * 1024L)  // 14 GiB


#define PI 3.141592653589793238462643383279502884f


#endif
