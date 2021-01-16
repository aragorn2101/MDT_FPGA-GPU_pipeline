#!/bin/bash

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
