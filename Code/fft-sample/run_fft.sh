#!/bin/bash
echo SERIAL:
./fft-ser audio-files/cat.wav serial_cat.csv

echo PARALLEL:
./fft-par audio-files/cat.wav cuda_cat.csv
echo

