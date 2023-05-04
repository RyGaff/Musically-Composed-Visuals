#!/bin/bash

#get the input wav file
inputwav=${1:?input name of wav file that is in /fft-sample/audio-files/ WITHOUT the .wav extension}

echo running $inputwav
echo
#Run the ffts
echo Running fft/dft:
echo SERIAL:
./fft-sample/fft-ser ./fft-sample/audio-files/$inputwav.wav serial_$inputwav.csv

echo PARALLEL:
./fft-sample/fft-par ./fft-sample/audio-files/$inputwav.wav parallel_$inputwav.csv

#Get the name of the file created

# Run serial or parallel visualizer based on arguments
echo Running Visualizers:
echo SERIAL:
./visuals/vis 0 normalized_parallel_$inputwav.csv

echo PARALLEL:
./visuals/par_vis 0 normalized_parallel_$inputwav.csv
