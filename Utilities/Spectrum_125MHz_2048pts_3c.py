#!/usr/bin/env python2
#
#  Script to download data from SNAP board and plot spectra.  Working bandwidth
#  is 125 MHz with a resolution of 2048 points and maximum number of input
#  analogue channels is 3.
#  Version 1.0
#
#  Copyright (C) 2020  Nitish Ragoomundun, Mauritius
#                      lrugratz gmail com
#                              @     .
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
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
#  IN THE SOFTWARE.
#


from sys import argv, stdout, exit
from time import sleep
from logging import StreamHandler, getLogger
from optparse import OptionParser
from os import path
from numpy import linspace, array, log10
from numpy import float as np_float
import pylab as plb
import matplotlib.pyplot as plt
from struct import unpack
import casperfpga, casperfpga.snapadc


###  Constants ###
freq_range_mhz = linspace(0, 125, 2048)


###  Functions  ###
def exit_fail():
    exit()

def exit_clean():
    try:
        for f in fpgas: f.stop()
    except: pass
    exit()


###  BEGIN Function to fetch data  ###
def fetch_data(spectrumId):
    # Fetch data from SNAP bram
    spectrum = unpack('>2048Q', fpga.read(spectrumId, 2048*8, 0))  # 8 bytes

    spectrumIdx = fpga.read_uint('acc_count')

    return spectrumIdx, array(spectrum, dtype=np_float)

###  END Function to fetch data  ###



###  BEGIN Function to animate all 3 channels  ###
def animate_3plots():
    # Analogue input port
    inputId = ["SMATP1","SMATP5","SMATP9"]

    # Plot line colour
    colorId = ["blue", "green", "purple"]

    # Clear plots
    plt.clf()
    ax = []

    for channelIdx in range(0,3):
        # Append a new subplot
        ax.append(fig.add_subplot(2,2,channelIdx+1))

        # Update data vectors
        spectrumIdx, freq_channels = fetch_data("spectrum_"+str(channelIdx))

        power = 10*log10(freq_channels)
        # Restore DC to a value within appropriate range
        power[0] = power[-1]

        ax[channelIdx].set_title("Spectrum for input at {:s}".format(inputId[channelIdx]))
        ax[channelIdx].set_xlabel("Frequency / MHz")
        ax[channelIdx].set_xlim(freq_range_mhz[0], freq_range_mhz[-1])
        ax[channelIdx].set_xticks(linspace(freq_range_mhz[0], freq_range_mhz[-1], 11))
        ax[channelIdx].set_ylabel("Power / dB")
        ax[channelIdx].grid(True)
        ax[channelIdx].plot(freq_range_mhz, power, linewidth=1, color=colorId[channelIdx])

    fig.canvas.draw()
    fig.canvas.manager.window.after(300, animate_3plots)
    plt.tight_layout()

###  END Function to animate all 3 channels  ###



###  BEGIN Function to animate channel on SMATP1  ###
def animate_SMATP1():
    # Clear plots
    plt.clf()

    # Append a new subplot
    ax = fig.add_subplot(1,1,1)

    # Fetch data from SNAP
    spectrumIdx, freq_channels = fetch_data("spectrum_0")

    power = 10*log10(freq_channels)
    # Restore DC to a value within appropriate range
    power[0] = power[-1]

    ax.set_title("Spectrum for input at SMATP1")
    ax.set_xlabel("Frequency / MHz")
    ax.set_xlim(freq_range_mhz[0], freq_range_mhz[-1])
    ax.set_xticks(linspace(freq_range_mhz[0], freq_range_mhz[-1], 11))
    ax.set_ylabel("Power / dB")
    ax.grid(True)
    ax.plot(freq_range_mhz, power, linewidth=1, color="blue")

    fig.canvas.draw()
    fig.canvas.manager.window.after(300, animate_SMATP1)
    plt.tight_layout()

###  END Function to animate channel on SMATP1  ###



###  BEGIN Function to animate channel on SMATP5  ###
def animate_SMATP5():
    # Clear plots
    plt.clf()

    # Append a new subplot
    ax = fig.add_subplot(1,1,1)

    # Fetch data from SNAP
    spectrumIdx, freq_channels = fetch_data("spectrum_1")

    power = 10*log10(freq_channels)
    # Restore DC to a value within appropriate range
    power[0] = power[-1]

    ax.set_title("Spectrum for input at SMATP5")
    ax.set_xlabel("Frequency / MHz")
    ax.set_xlim(freq_range_mhz[0], freq_range_mhz[-1])
    ax.set_xticks(linspace(freq_range_mhz[0], freq_range_mhz[-1], 11))
    ax.set_ylabel("Power / dB")
    ax.grid(True)
    ax.plot(freq_range_mhz, power, linewidth=1, color="green")

    fig.canvas.draw()
    fig.canvas.manager.window.after(300, animate_SMATP5)
    plt.tight_layout()

###  END Function to animate channel on SMATP5  ###



###  BEGIN Function to animate channel on SMATP9  ###
def animate_SMATP9():
    # Clear plots
    plt.clf()

    # Append a new subplot
    ax = fig.add_subplot(1,1,1)

    # Fetch data from SNAP
    spectrumIdx, freq_channels = fetch_data("spectrum_2")

    power = 10*log10(freq_channels)
    # Restore DC to a value within appropriate range
    power[0] = power[-1]

    ax.set_title("Spectrum for input at SMATP9")
    ax.set_xlabel("Frequency / MHz")
    ax.set_xlim(freq_range_mhz[0], freq_range_mhz[-1])
    ax.set_xticks(linspace(freq_range_mhz[0], freq_range_mhz[-1], 11))
    ax.set_ylabel("Power / dB")
    ax.grid(True)
    ax.plot(freq_range_mhz, power, linewidth=1, color="purple")

    fig.canvas.draw()
    fig.canvas.manager.window.after(300, animate_SMATP9)
    plt.tight_layout()

###  END Function to animate channel on SMATP9  ###



###  BEGIN Parsing command line arguments  ###
parser = OptionParser(usage = "%prog [OPTIONS] <SNAP_IP_ADDRESS>")

parser.add_option("--no-program",
                  dest = "noprogram",
                  action = "store_true",
                  help = "Skip reprogramming the FPGA.")

parser.add_option("-b", "--fpgfile",
                  dest = "fpgfile",
                  type = "string",
                  default = "",
                  help = "Specify fpg file to load.")

parser.add_option("-c", "--channel",
                  dest = "input_mode",
                  type = "int",
                  default = 0,
                  help = "Switches to single channel mode and parameter indicates which SMATP port is used for input. Valid range of input is [1, 5, 9].")

(opts, args) = parser.parse_args()

###  END Parsing command line arguments  ###



###  BEGIN Main program  ###
try:
    # Verify if a SNAP IP was entered
    if (args == []):
        raise ValueError

    # Verify if a fpg file was given and if path is correct
    if (not opts.noprogram and not path.isfile(opts.fpgfile)):
        raise IOError

    # Verify if input mode is well set (1 or 3 channel(s) mode)
    if (opts.input_mode != 0):
        if (opts.input_mode not in [1, 5, 9]):
            print("Input analogue channel can only be in set {1, 5, 9}.")
            raise ValueError


    # SNAP board at
    snap_Rpi_IP = args[0]

    # Activate logging
    logger = getLogger(snap_Rpi_IP)
    logger.addHandler(StreamHandler())
    logger.setLevel(10)

    # Establish connection and create fpga instance
    print("Connecting to SNAP board via Raspberry Pi @{:s} ...".format(snap_Rpi_IP))
    fpga = casperfpga.CasperFpga(snap_Rpi_IP, logger=logger)
    print("Done.")
    sleep(1)

    if fpga.is_connected():
        print("Connection established.")
    else:
        print("ERROR connecting to SNAP @{:s}.".format(snap_Rpi_IP))
        exit_fail()


    # Programming FPGA on SNAP
    print("\n-------------------------------------------------------")
    stdout.flush()
    if not opts.noprogram:
        print("Programming FPGA with file {:s} ...".format(opts.fpgfile))
        fpga.upload_to_ram_and_program(opts.fpgfile)
        print("Done.")
        sleep(5)
    else:
        print("Skipped re-programming FPGA.")


    # We need to configure the ADC chips. The following function call assumes
    # that the SNAP has a 10 MHz reference input connected. It will use this
    # reference to generate a 250 MHz sampling clock. The init function will
    # also tweak the alignment of the digital lanes that carry data from the
    # ADC chips to the FPGA, to ensure reliable data capture. It should take
    # about 30 seconds to run.
    adc = casperfpga.snapadc.SNAPADC(fpga, ref=10) # reference at 10MHz

    # We want a sampling rate of 250 Mhz, with 4 channels per ADC chip, using 8-bit ADCs
    print("\n-------------------------------------------------------")
    stdout.flush()
    if not opts.noprogram:
        print("Attempting to initialize ADC chips ...")

        # Try initializing a few times for good measure in case it fails.
        adc_done = False
        for i in range(3):
            if ( adc.init(samplingRate=250, numChannel=4, resolution=8) == 0 ):
                adc_done = True
                break

        if not adc_done:
            print("Failed to calibrate ADC after {:d} attempt(s).!".format(i+1))
            exit_clean()
        else:
            print("Done with initialisation. Took {:d} attempt(s)".format(i+1))

    # Since we're in 4-way interleaving mode (i.e., one input per ADC) we
    # should configure the ADC inputs accordingly
    adc.selectADC(0)  # send commands to the first ADC chip
    adc.adc.selectInput([1,2,3,4])  # Interleave four ADCs with first lane pointing to SMATP1
    adc.selectADC(1)  # send commands to the first ADC chip
    adc.adc.selectInput([1,2,3,4])  # Interleave four ADCs with first lane pointing to SMATP5
    adc.selectADC(2)  # send commands to the first ADC chip
    adc.adc.selectInput([1,2,3,4])  # Interleave four ADCs with first lane pointing to SMATP9


    print("\n-------------------------------------------------------")
    print("Reset counters ...")
    stdout.flush()
    fpga.write_int('counter_rst', 1)
    fpga.write_int('counter_rst', 0)
    print("Done.")


    # sleep 2 seconds
    sleep(2)

    # Set up figure
    print("\n-------------------------------------------------------")
    fig = plt.figure()

    if (opts.input_mode == 0):
        print("Generating plots ...")
        fig.canvas.manager.window.after(300, animate_3plots())
    else:
        print("Generating plot ...")
        if (opts.input_mode == 1):
            fig.canvas.manager.window.after(300, animate_SMATP1())
        elif (opts.input_mode == 5):
            fig.canvas.manager.window.after(300, animate_SMATP5())
        elif (opts.input_mode == 9):
            fig.canvas.manager.window.after(300, animate_SMATP9())

    plt.show()


except KeyboardInterrupt:
    exit_clean()
except ValueError:
    print("Usage: {:s} [OPTIONS] <SNAP_IP_ADDRESS>".format(argv[0]))
    print("Please run with the -h flag to see all options.")
    exit_fail()
except IOError:
    print("Cannot access fpgfile!")
    exit_fail()
except Exception as inst:
    print type(inst)
    print inst.args
    print inst
    exit_fail()

###  END Main program  ###


exit_clean()
