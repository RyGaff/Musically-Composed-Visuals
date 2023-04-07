CFLAGS= -g -Wall
CC    = g++
TARGET= main-ser main-par

all: $(TARGET)

main-ser: main.cpp
	$(CC) $(CFLAGS) -Wno-unknown-pragmas -o main-ser main.cpp
	
main-par: main.cpp
	$(CC) $(CFLAGS) -fopenmp -o main-par main.cpp

clean:
	rm -f $(TARGET)
