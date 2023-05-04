CFLAGS= -g -Wall
CC    = g++
TARGET= clean fft-ser fft-par 

all: $(TARGET)

fft-ser: fft.cpp
	$(CC) $(CFLAGS) -Wno-unknown-pragmas -o fft-ser fft.cpp
	
fft-par: fft.cpp
	nvcc -o fft-par fft.cu

clean:
	rm -f $(TARGET) *.csv
