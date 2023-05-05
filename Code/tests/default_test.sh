#!/bin/bash

# Small little test file that tests the few wav files we were able to add to project directory 

cd ..
./run.sh cat
cd tests/

cd ..
./run.sh humb
cd tests/

clear

# CAT TESTS
echo DFT DIFF CAT:
diff -y --suppress-common-lines ../parallel_cat_dft.csv ../serial_cat_dft.csv | grep '^' | wc -l 

echo DFT NORMALIZED DIFF CAT:
diff -y --suppress-common-lines ../normalized_parallel_cat_dft.csv ../normalized_serial_cat_dft.csv | grep '^' | wc -l 

echo FFT DIFF CAT:
diff -y --suppress-common-lines ../parallel_cat.csv ../serial_cat.csv | grep '^' | wc -l 

echo FFT NORMALIZED DIFF CAT:
diff -y --suppress-common-lines ../normalized_parallel_cat.csv ../normalized_serial_cat.csv | grep '^' | wc -l

# HUMB TESTS
echo DFT DIFF HUMB:
diff -y --suppress-common-lines ../parallel_humb_dft.csv ../serial_humb_dft.csv | grep '^' | wc -l

echo DFT NORMALIZED DIFF HUMB:
diff -y --suppress-common-lines ../normalized_parallel_humb_dft.csv ../normalized_serial_humb_dft.csv | grep '^' | wc -l

echo FFT DIFF HUMB:
diff -y --suppress-common-lines ../parallel_humb.csv ../serial_humb.csv | grep '^' | wc -l

echo FFT NORMALIZED DIFF HUMB:
diff -y --suppress-common-lines ../normalized_parallel_humb.csv ../normalized_serial_humb.csv | grep '^' | wc -l