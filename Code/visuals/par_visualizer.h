#include "GPU_Animation_Bitmap.h"
#include "dependencies.h"

#define block 1024

void key_listener(unsigned char key, int x, int y);
void arrow_listener(int key, int x, int y);
void print_stats(float render_time);
double *csv_to_array(FILE *fp);
