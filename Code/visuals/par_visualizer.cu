#include "par_visualizer.h"
// Init our animation bitmap.
// This is where most of the opengl and cuda interop stuff is.
GPUAnimBitmap *bitmap_Ptr;

// Default starting values for cRe and cIm
// Just our default starting values
float const REAL = -0.7;
float const IMAGINARY = 0.27015;
// float const REAL = -0.773760;
// float const IMAGINARY = 0.112774;

// Variables that modify the function/user controls
float cRe = REAL;
float cIm = IMAGINARY;
int max_iterations = 250;
double zoom = 1;
double mx = 0;
double my = 0;
int animation = 0;

// Global helper variables
int Step_To_Seek = 0;
unsigned int frames = 0;
char *read_from_file;
double t = 0.0;
FILE *fp;
int charcount = 0;
char *stop;
double mintime = 99999;

__global__ void julia(uchar4 *pixels, int max_iterations,
                      double cRe, double cIm, double mX, double mY, double zoom)
{
    double zx, zy, ox, oy;
    int height = block;
    int width = block;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    zx = 1.5 * (x - width / 2) / (0.5 * zoom * width) + mX;
    zy = (y - height / 2) / (0.5 * zoom * height) + mY;

    int iteration = 0;
    float magnitude = 0.0;
    while (iteration < max_iterations)
    {
        ox = zx;
        oy = zy;
        zx = (ox * ox - oy * oy) + cRe;
        zy = (ox * oy + ox * oy) + cIm;
        magnitude = zx * zx + zy * zy;
        if ((magnitude) > 4) break;   
        iteration++;
    }

    if (iteration == max_iterations)
    { 
        pixels[offset].x = 255 * magnitude/2;
        pixels[offset].y = 255 * magnitude/4;
        pixels[offset].z = 255;
        pixels[offset].w = 255;

    } else {
        pixels[offset].x = 0;
        pixels[offset].y = (255 * iteration/max_iterations)/magnitude;
        pixels[offset].z = 255 * iteration/max_iterations;
        pixels[offset].w = 255;
    }
}

void generateFrame(uchar4 *ptr)
{
    dim3 grids(block / 16, block / 16);
    dim3 threads(16, 16);
    if (animation == 1)
    {
        double old_time = t;
        t = clock();
        double delta_time = (t - old_time)/CLOCKS_PER_SEC;
        double *buf;
        buf = csv_to_array(fp);
    
        cRe =  (cRe + (.001 * atan(buf[1]))) + 0.001 *  tan(delta_time);
        cIm =  (cIm + (.01  * atan(buf[0]))) + 0.0001 * tan(delta_time); 

    } 

    START_TIMER(julia);
    julia<<<grids, threads>>>(ptr, max_iterations, cRe, cIm, mx, my, zoom);
    cudaDeviceSynchronize();
    STOP_TIMER(julia);
    
    if (GET_TIMER(julia) < mintime){
        mintime = GET_TIMER(julia);
    }

    if(strcmp(stop, "1") != 0){
        if (frames == 50){
            printf("Min Frame time for parllel version %lf\n", mintime);
            bitmap_Ptr->free_resources();
            exit(0);
        } else {
            printf("Remaining frames to generate %d\t\tmintime = %lf\n", 50 - frames++, mintime);
        }
    } 
//    print_stats(GET_TIMER(julia));
}

int main(int argc, char* argv[])
{
    if (argc != 3){
        printf("Usage: ./par_vis <1 for continuous> <csv file>");   
        printf("if the first argument is anything else the program stops after 50 frames are generated");
        exit(1);
    }

    stop = argv[1];

    if (!fp){
        fp = fopen(argv[2], "r");
        if (!fp)
        {
            puts("Failed to open file");
            exit(EXIT_FAILURE);
        }
        for (char c = getc(fp); c != EOF; c = getc(fp)){
            charcount++;        
        }

        fseek(fp, 0, SEEK_SET);
    }

    GPUAnimBitmap bitmap(block, block, NULL);
    bitmap_Ptr = &bitmap;

    t = time(0);
    // File and resources are cleaned up in the exit function located in the header file
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
        printf("Remaining frames to generate %d\t\tmintime = %lf\n", 50 - frames++, mintime);
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

double *csv_to_array(FILE *fp)
{
    static double ret[2];

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

    Step_To_Seek = (len + Step_To_Seek) % charcount;

    return ret;
}
