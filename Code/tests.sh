
#!/bin/bash

#get the input wav file
inputwav=$1
output=test_results.txt
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
echo Starting iterations tests
for iterations in 50 100 250 500 1000 2500 5000 10000:
do
echo SERIAL--$iterations:
echo SERIAL--$iterations: >> serial_$output
./visuals/vis 0 normalized_parallel_$inputwav.csv $iterations >> serial_$output

echo PARALLEL--$iterations:
echo PARALLEL--$iterations: >> parallel_$output
./visuals/par_vis 0 normalized_parallel_$inputwav.csv $iterations >> parallel_$output
done

echo Finished running tests
