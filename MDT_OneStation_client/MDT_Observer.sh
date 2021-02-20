#!/bin/bash
#
#  MDT_Observer.h
#
#  Script to execute the multi-threaded parallel client program on the
#  GPU processing node, according to the configuration set up for the
#  instrument in file MDT_SystemParams.conf
#
#  Copyright (c) 2021 Nitish Ragoomundun
#                    <lrugratz gmail com>
#                             @     .
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to
#  deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
#  sell copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#


# Source MDT instrument configuration file, which should be in the current
# directory.
. ./MDT_SystemParams.conf


set -e

# First check if the script exists and is executable
if [ ! -e ${MDTSERVER} ]; then
  echo "Cannot access ${MDTSERVER}!"
  exit 1
elif [ ! -x ${MDTSERVER} ]; then
  echo "${MDTSERVER} is not executable!"
  exit 2
fi


echo "---------------------------------------"
echo "Executing the following command:"
echo "${MPIEXEC} -np ${NUMTASKS} ${MDTSERVER} ${NELEMENTS} ${NPOLS} ${NCHANNELS} ${NTAPS} ${NSPECTRA} ${SPERELEMENT}"
echo "---------------------------------------"
echo
${MPIEXEC} -np ${NUMTASKS} ${MDTSERVER} ${NELEMENTS} ${NPOLS} ${NCHANNELS} ${NTAPS} ${NSPECTRA} ${SPERELEMENT}


exit 0
