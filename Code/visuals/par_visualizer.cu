#include "GPU_Animation_Bitmap.h"
#include "dependencies.h"

#define block 1024

// Init our animation bitmap.
// This is where most of the opengl and cuda interop stuff is.
GPUAnimBitmap *bitmap_Ptr;

// Default starting values for cRe and cIm
// Just our default starting values
float const REAL = -0.7;
float const IMAGINARY = 0.27015;

// Variables that modify the function/user controls
float cRe = REAL;
float cIm = IMAGINARY;
int max_iterations = 150;
double zoom = 1;
double mx = 0;
double my = 0;
int animation = 0;

// Global helper variables
int Step_To_Seek = 0;
unsigned int frames = 0;
double t = 0.0;
char read_from_file[] = "normalized_normTest.csv";

void key_listener(unsigned char key, int x, int y);
void arrow_listener(int key, int x, int y);
void print_stats(float render_time);
double *csv_to_array(char *file);

__global__ void julia(uchar4 *pixels, int max_iterations,
                      double cRe, double cIm, double mX, double mY, double zoom)
{
    double zx, zy, ox, oy;
    int height = block;
    int width = block;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    zx = 1.5 * (x - width / 2) / (0.5 * zoom * width) - mX;
    zy = (y - height / 2) / (0.5 * zoom * height) + mY;

    int iteration = 0;
    while (iteration < max_iterations)
    {
        ox = zx;
        oy = zy;
        zx = (ox * ox - oy * oy) + cRe;
        zy = (ox * oy + ox * oy) + cIm;
        if ((zx * zx + zy * zy) > 4)
        {
            pixels[offset].x = 0.0;
            pixels[offset].y = 0.0;
            pixels[offset].z = 0.0;
            pixels[offset].w = 255;
            return;
        }
        iteration++;
    }

    if (iteration == max_iterations)
    {
        pixels[offset].x = (unsigned char)255 * 0;
        pixels[offset].y = (unsigned char)255 * (ox * ox);
        pixels[offset].z = (unsigned char)255 * oy;
        pixels[offset].w = 255;
    }
}

void generateFrame(uchar4 *ptr)
{
    dim3 grids(block / 16, block / 16);
    dim3 threads(16, 16);
    if (animation == 1)
    {

        double *buf;
        buf = csv_to_array(read_from_file);

        double old_time = t;
        // t = gettimeofday(&tv, NULL);
        t = clock();
        double delta_time = (t - old_time);

        // cRe = (cRe + buf[0]) +  0.000001 * sin(delta_time/zoom);
        cRe = (cRe) +  0.01 * (buf[0] * sin(delta_time));
        // cIm = (cIm + buf[1]) +  0.000001 * cos(delta_time/zoom); 
        cIm = (cIm) +  0.01 * (buf[1] * cos(delta_time));
    }

    START_TIMER(julia);
    julia<<<grids, threads>>>(ptr, max_iterations, cRe, cIm, mx, my, zoom);
    cudaDeviceSynchronize();
    STOP_TIMER(julia);

    print_stats(GET_TIMER(julia));
    frames++;
}

int main(void)
{
    GPUAnimBitmap bitmap(block, block, NULL);
    bitmap_Ptr = &bitmap;

    bitmap.anim_and_exit((void (*)(uchar4 *, void *))generateFrame, NULL);
    glutKeyboardFunc(key_listener);
    glutSpecialFunc(arrow_listener);
    glutMainLoop();
}

void key_listener(unsigned char key, int x, int y)
{
    switch (key)
    {
    case 'q':
        // Free our buffers and cuda mem
        bitmap_Ptr->free_resources();
        exit(0);
        break;
    case 'w':
        max_iterations += 10;
        break;
    case 's':
        if (max_iterations > 0)
        {
            max_iterations -= 10;
        }
        break;
    // Adjust real
    case 'r':
        cRe += .1;
        break;
    case 'f':
        cRe -= .1;
        break;
    // Adjust Imaginary
    case 'i':
        cIm += .01;
        break;
    case 'k':
        cIm -= .01;
        break;
    case 32: // Spacebar / Toggle animation
        if (animation == 0)
        {
            animation = 1;
        }
        else
        {
            animation = 0;
        }
        break;
    case '=': // Zoom Camera In
        zoom += .1;
        break;
    case '-': // Zoom Camera Out
        zoom -= .1;
        break;
    default:
        // printf("Key id_%d is not a valid input\n", key);
        printf("Valid keys:\n\
            q = exit\n\
            w = increment iterations by 10\n\
            s = decrement iterations by 10\n\
            r = increment real by .1\n\
            s = decrement real by .1 \n\
            i = increment imaginary by .01\n\
            k = decrement imaginary by .01\n\
            space = enable/disable animation\n\
            = = zoom in\n - = zoom out\n\
            ArrowKeys to pan camera in a direction\n\
            ");
        return;
    }
}

void arrow_listener(int key, int x, int y)
{
    switch (key)
    {
    case GLUT_KEY_LEFT:
        mx -= .1;
        break;
    case GLUT_KEY_RIGHT:
        mx += .1;
        break;
    case GLUT_KEY_UP:
        my += .1;
        break;
    case GLUT_KEY_DOWN:
        my -= .1;
        break;
    }
}

void print_stats(float render_time)
{
    printf("Time To Render = %f:\n", render_time);
    printf("Iterations %d   real = %f   imaginary = %f   animation = %d\n", max_iterations, cRe, cIm, animation);
    printf("    Camera: pos = %f, %f   zoom = %f\n", mx, my, zoom);
}

double *csv_to_array(char *file)
{
    static double ret[2];
    FILE *fp;
    fp = fopen(file, "r");
    if (!fp)
    {
        puts("File Not Found");
        exit(EXIT_FAILURE);
    }

    fseek(fp, Step_To_Seek, SEEK_SET);
    char *token;
    char buffer[200];

    fgets(buffer, 200, fp);
    token = strtok(buffer, ",");
    int i = 0;
    int len = 0;
    while (token != NULL)
    {
        len += strlen(token);
        ret[i] = strtod(token, &token);
        token = strtok(NULL, ",");
        i += 1;
    }

    Step_To_Seek = (len + Step_To_Seek) % 4126;

    return ret;
}
