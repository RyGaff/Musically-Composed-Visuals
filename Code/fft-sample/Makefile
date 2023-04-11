CFLAGS= -g -Wall
CC    = g++
TARGET= main-ser main-par

all: $(TARGET)

main-ser: fft.cpp
	$(CC) $(CFLAGS) -Wno-unknown-pragmas -o fft-ser fft.cpp
	
main-par: fft.cpp
	$(CC) $(CFLAGS) -fopenmp -o fft-par fft.cpp

clean:
	rm -f $(TARGET)
