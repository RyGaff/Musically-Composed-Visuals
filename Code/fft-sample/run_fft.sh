#!/bin/bash
echo SERIAL:
./fft-ser audio-files/cat.wav serial_cat.csv
echo PARALLEL:
OMP_NUM_THREADS=$t ./fft-par audio-files/cat.wav cuda_cat.csv
echo

