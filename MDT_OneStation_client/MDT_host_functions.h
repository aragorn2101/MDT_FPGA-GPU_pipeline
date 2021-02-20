/*
 *  MDT_host_functions.h
 *
 *  Header file for the host functions.
 *
 *  Copyright (c) 2021 Nitish Ragoomundun
 *                    <lrugratz gmail com>
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
void *get_address(struct sockaddr *);


#endif
