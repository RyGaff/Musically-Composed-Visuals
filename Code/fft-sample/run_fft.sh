#!/bin/bash
echo SERIAL:
./fft-ser audio-files/cat.wav serial_cat.csv
echo PARALLEL:
for t in 1 2 4 8 16 32; do
    OMP_NUM_THREADS=$t ./fft-par audio-files/cat.wav "$t"_parallel_cat.csv
done
echo

