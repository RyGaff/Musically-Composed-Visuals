#include "dependencies.h"
#include "gl_helper.h"
#include "cuda_gl_interop.h"

#define block 1024

void display();
void animate();
void julia(double zoom, double mX, double mY);
void key_listener(unsigned char key, int x, int y);
void arrow_listener(int key, int x, int y);
void print_stats(float render_time);
double* csv_to_array(); 
int driver(int argc, char** argv);
