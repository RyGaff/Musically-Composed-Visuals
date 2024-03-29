#!/bin/bash

#get the input wav file
inputwav=$1

cd fft-sample/
make
cd ..
cd visuals/
make
cd ..

echo $inputwav
#Run the ffts
echo Running fft/dft:
echo SERIAL:
./fft-sample/fft-ser ./fft-sample/audio-files/$inputwav.wav serial_$inputwav.csv

echo PARALLEL:
./fft-sample/fft-par ./fft-sample/audio-files/$inputwav.wav parallel_$inputwav.csv

#Get the name of the file created

# Run serial or parallel visualizer based on arguments
echo Running 50 frames of visualizers:
echo SERIAL:
./visuals/vis 0 normalized_parallel_$inputwav.csv

echo PARALLEL:
./visuals/par_vis 0 normalized_parallel_$inputwav.csv

echo Display visualizers allowing for user control

echo Running the serial version
./visuals/vis 1 normalized_parallel_$inputwav.csv

echo Running the Parallel version
./visuals/par_vis 1 normalized_parallel_$inputwav.csv

