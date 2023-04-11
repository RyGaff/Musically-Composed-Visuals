/**
 * CS470 - Final Project - Audio Visuals Team
 *
 * Audio Processor Program
 *
 * Team: Matthew Dim, Ryan Gaffney, Ian Lips, Justin Choi
 */
#include <cstdint>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <fstream>
#include <bits/stdc++.h>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "timer.h"
using namespace std;

typedef complex<double> cd;

struct WAV_HEADER{
    // Riff
    uint8_t chunkID[4];
    uint32_t chunkSize;
    uint8_t format[4];
    // FMT
    uint8_t subchunk1ID[4];
    uint32_t subchunk1Size;
    uint16_t audioFormat;
    uint16_t numChannels;
    uint32_t samepleRate;
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample;
    // Data
    uint8_t subchunk2ID[4];
    uint32_t subchunk2Size;
};

int thread_count;

vector<complex<double>> convertWavDataToComplexVector(vector<int16_t> d){
    vector<complex<double>> output;
    for(int16_t i : d){
        output.push_back(complex<double>(i,0));
    }
    return output;
}

vector<complex<double>> getDataFromWav(const std::string &file_path){
    ifstream wav(file_path);
    struct WAV_HEADER whr;

    if(!wav.is_open()){
        throw "File Could Not be Opened!";
    }
    
    // Riff
    wav.read((char*)&whr.chunkID, sizeof(whr.chunkID)); 
    wav.read((char*)&whr.chunkSize, sizeof(whr.chunkSize));
    wav.read((char*)&whr.format, sizeof(whr.format));

    // fmt
    wav.read((char*)&whr.subchunk1ID, sizeof(whr.subchunk1ID));
    wav.read((char*)&whr.subchunk1Size, sizeof(whr.subchunk1Size));
    wav.read((char*)&whr.audioFormat,sizeof(whr.audioFormat));
    wav.read((char*)&whr.numChannels,sizeof(whr.numChannels));
    wav.read((char*)&whr.samepleRate,sizeof(whr.samepleRate));
    wav.read((char*)&whr.byteRate,sizeof(whr.byteRate));
    wav.read((char*)&whr.blockAlign,sizeof(whr.blockAlign));
    wav.read((char*)&whr.bitsPerSample,sizeof(whr.bitsPerSample));

    // Data
    wav.read((char*)&whr.subchunk2ID,sizeof(whr.subchunk2ID));
    wav.read((char*)&whr.subchunk2Size,sizeof(whr.subchunk2Size));

    vector<int16_t> data(whr.subchunk2Size);
    
    for(uint32_t i = 0; i < whr.subchunk2Size; i++){
        wav.read((char*)&data[i],sizeof(data[i]));
    }

    // Read audio data
    wav.close();
    
    return convertWavDataToComplexVector(data);
}

void dft(vector<complex<double>> signal,vector<complex<double>>& output){
    for(uint64_t k = 0; k < signal.size(); k++){
        complex<double> ans(0,0);
        for(uint64_t t = 0; t < signal.size(); t++){
            double angle = (-2 * M_PI * t * k) / signal.size(); 	
            ans += signal[t] * exp(complex<double>(0,angle));
        }
        output.push_back(ans);
    }
}


/*
 * Power of 2 helper function.
 */
constexpr int findNextPowerOfTwo(int N){
    N--;
    N |= N >> 1;
    N |= N >> 2;
    N |= N >> 4;
    N |= N >> 8;
    N |= N >> 16;
    N++;
    return N;
}

/*
 * Power of 2 helper function.
 */
constexpr bool isPowerOfTwo(int N){
    return (N & (N-1)) == 0;
}

/*
 * Ensure that signal length is a power of two
 */
void transformSignal(vector<complex<double>>& signal){
    int diff = isPowerOfTwo(signal.size()) ? 0 : findNextPowerOfTwo(signal.size()) - signal.size();
    if(diff == 0) {
        return;
    }else{
        for(int i = 0; i < diff; i++){
            signal.push_back(0);	
        }
    }
}


/*
 * Perform a Fast Fourier Transform (FFT) on audio data.
 *
 * Output will be a vector.
 */
void fft(vector<complex<double>>& signal){
// #  pragma omp parallel default(none) shared(signal)
    {
   
        // Thread count
#       ifdef _OPENMP
        thread_count = omp_get_thread_num();
        printf("Thread Count Set At: %d", thread_count);
#       else
        thread_count = 1;
#       endif

        transformSignal(signal);
        int N = signal.size();

        if(N == 1) return; // Base case

        vector<complex<double>> even(N/2), odd(N/2); 
// #       pragma omp for
        for(int i = 0; 2 * i < N; i++){
            even[i] = signal[2*i];
            odd[i] = signal[2*i+1];
        }

        fft(even);
        fft(odd);

        double angle = 2 * M_PI / N;
        complex<double> w_n(cos(angle),sin(angle));
        complex<double> w(1);
//#       pragma omp for shared(even, odd) reduction(*:w)
        for(int i = 0; 2 * i < N; i++){
             signal[i] = even[i] + w * odd[i];
             signal[i + N/2] = even[i] - w * odd[i];
             w *= w_n;
        } 
    }
}

// https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
// cd is a complex double
/*
 * Bit reversal algorithm for the iterative version of fft.
 * this is needed because we are doing a bottom up implementation isntead of 
 * top down like we did with the recursive fft
 *  
 */
unsigned int bit_reversal(unsigned int i, int log2n){
    int rev = 0;
    for (int j = 0; j < log2n; j++) {
        rev <<= 1;
        rev |= (i & 1);
        i >>= 1;
    }
    return rev;
}
/*
 * Recursion? who even is she.
 * a - we read from here
 * A - this will be our new output
 * log2n - the number of "merges" if you will. 
 *         The logic is similar to the parallel merge sort stuff with the bitwise operations.
 */
void iterative_fft(vector<cd>& a, vector<cd>& A, int log2n){
    int n = a.size();
    unsigned un = a.size(); // Done to stop a warning about comparing unsigned an signed ints, shouldn't change functionality

    for (unsigned int i = 0; i < un; ++i) {
        A[i] = a[bit_reversal(i, log2n)];
    }
 
    const complex<double> J(0, 1);
    for (int s = 1; s <= log2n; ++s) {
        int m = 1 << s; // 2 power s
        int m2 = m >> 1; // m2 = m/2 -1
        cd w(1, 0);
 
        cd wm = exp(J * (3.1415926536 / m2));
        for (int j = 0; j < m2; ++j) {
            for (int k = j; k < n; k += m) {
                cd t = w * A[k + m2]; 
                cd u = A[k];
                A[k] = u + t;
                A[k + m2] = u - t;
            }

            w *= wm;
        }
    }
}

/*
 * Python helper function to plot fourier transform
 */
void plotOutputData(){
    system("python3 ./python-stuffs/plotter.py");
}


/*
 * Write data to a CSV file
 *
 * File will be parsed in visualizer
 * 
 * Return: The number of complex numbers ie number of lines, will make wrapping with the visalizer easier
 */
int writeDataToCSVFile(const vector<complex<double>>& out, const string fileName = "coords.csv"){

    ofstream outFile(fileName);
    outFile << "x,y" << "\n";
    int count = 0;
    for(complex<double> i : out){
        count++;
        outFile << i.real() << "," << i.imag() << "\n";
    }

    outFile.close();

    // plotOutputData();
    return count;
}

int main(int argc,const char** argv){

    cin.tie(0);

    // Read input file
    if (argc != 3) {
   	printf("Usage: %s <input-wav-file-name> <output-csv-file-name>\n", argv[0]); 
	exit(EXIT_FAILURE);
    }

    std::string file_name = argv[1];
    std::string csv_name  = argv[2];

    vector<complex<double>> output = getDataFromWav(file_name);     
    // Add a timer to test for parallelism
    // Add a barrier if needed
    transformSignal(output); // Ensure that output size is a power of 2
    vector<complex<double>> iterative_out(output.size());
    START_TIMER(fft);
    // fft(output);
    iterative_fft(output, iterative_out, log2(output.size()));

    STOP_TIMER(fft);
    
    // Serial for now - TODO: Add OMP def
    printf("Thread Count: %d - FFT Type: Misc for now - FFT Time: %lfs\n", thread_count, GET_TIMER(fft));
    writeDataToCSVFile(iterative_out, csv_name);
    // writeDataToCSVFile(output, "recursive_" + csv_name);

    return EXIT_SUCCESS;
}
