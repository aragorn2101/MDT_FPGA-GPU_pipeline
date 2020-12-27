# MDT FPGA-GPU pipeline

This repository is meant for sharing the software developed for the digital
pipeline of the Deuterium Telescope (Mauritius), commonly known as MDT. The
hardware for the digital back-end consists of parallel computing platforms:
FPGA, multi-core CPU and GPU.


## FPGA

The FPGA directory contains the Simulink design for the SNAP FPGA 10-channel
DAQ system for the MDT. A station of the MDT is equipped with 4 SNAP boards
running the design.

The pdf file is a snapshot of the design and note that it is much larger than
the A4 format. It is known that certain pdf reading software have trouble to
open it. Some software do open it but does not allow sufficient zoom so that
the small details can be seen. The software which are known to work are Okular
(on KDE) and xpdf.

The FPGA bit file was made by compiling the design with the CASPER Toolflow and
Vivado. The CASPER Toolflow is very sensitive when it comes to versioning of
every software used along the compilation pipeline. Thus, below are the
software versions we used.

- Ubuntu 16.04 LTS
- Xilinx Vivado SDK 2019.1.1
- Matlab R2018a with Simulink
- CASPER Toolflow (https://github.com/casper-astro/mlib_devel)
           commit: 09c2d3b27d02ffc65bf0b3d1954df4f5af62c6db with Python 3.5.2
- casperfpga (https://github.com/casper-astro/casperfpga)
           commit: ee9c43f2c066002c018741df9604aa751f413e69 with Python 2.7.16


## GPU

The directory contains source code for the GPU kernels used for the MDT FX
correlator. The correlator is the sequence of the following calls:

 1. ``PpS_Batch()`` kernel
 2. cuFFT library routine: ``cufftExecC2C()``
 3. ``ReorderXInput()`` kernel
 4. cuBLAS library kernel: ``cublasCgemmBatched( )``

The F-engine directory regroups the different implementations attempted for the
``PpS_Batch()`` kernel. The latter is a custom kernel which computes the
polyphase structure of the PFB before the Fourier transform.

The X-engine directory contains the two strategies implemented for the custom
``ReorderXInput()`` function, which re-orders the data in the format expected
by the cuBLAS kernel ``cublasCgemmBatched()``.


## Utilities

### Spectrum_\*.py, spectrum_\*.slx, spectrum_\*.fpg

The .slx file is the SNAP FPGA design file to implement data acquisition from 3 channels and a simple PFB to obtain a spectrum. The Python 2 script is the corresponding control script. Below is an example:

```
$ ./Spectrum_125MHz_2048pts_3c.py -h
Usage: Spectrum_125MHz_2048pts_3c.py [OPTIONS] <SNAP_IP_ADDRESS>

Options:
  -h, --help            show this help message and exit
  --no-program          Skip reprogramming the FPGA.
  -b FPGFILE, --fpgfile=FPGFILE
                        Specify fpg file to load.
  -c INPUT_MODE, --channel=INPUT_MODE
                        Switches to single channel mode and parameter
                        indicates which SMATP port is used for input. Valid
                        range of input is [1, 5, 9].

###  Example run  ###
$ ./Spectrum_125MHz_2048pts_3c.py -b spectrum_125mhz_2048pts_3c.fpg 172.22.37.221

# spectrum_125mhz_2048pts_3c.fpg samples at 250 MSps (Nyquist frequency) to
# obtain a 125 MHz spectrum subsequently.
```

**NOTE:**  FFT is done using 2048 points and analogue inputs are taken from
SMATP1, SMATP5 and SMATP9. A 10 MHz input clock is required at input SMATP14.

