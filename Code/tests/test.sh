#!/bin/bash

echo First File:
read firstCSV

echo Second File:
read secondCSV

diff -y --suppress-common-lines $firstCSV $secondCSV | grep '^' | wc -l