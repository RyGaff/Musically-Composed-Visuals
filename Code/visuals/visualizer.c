#include "dependencies.h"
#include "gl_helper.h"
#include "cuda_gl_interop.h"

#define block 1024

// Array to hold our pixels. To be used with glDrawPixels
float *pixels;

// Default starting values for cRe and cIm
// Just our default starting values
float const REAL = -0.7;
float const IMAGINARY = 0.27015;

// Dimensions of the window are set in the display function
int const height = block;
int const width = block;

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
char read_from_file[] = "normalized_normTest.csv";
double t = 0.0;

void display();
void animate();
void julia(double zoom, double mX, double mY);
void key_listener(unsigned char key, int x, int y);
void arrow_listener(int key, int x, int y);
void print_stats(float render_time);
double* csv_to_array(char* file); 

int main( int argc, char** argv )
{
    pixels = calloc(width * height * 4, sizeof(pixels));
    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize( width, height);
    glutInitWindowPosition(950,0);
    glutCreateWindow( "Julia" );

    glutDisplayFunc(animate);
    glutIdleFunc(animate);
    glutKeyboardFunc(key_listener);
    glutSpecialFunc(arrow_listener);
    glutMainLoop();
    free(pixels);
    return 0;
}

void julia(double zoom, double mX, double mY)
{
	double zx, zy, ox, oy;
    // algorithm to draw the julia set. 
    // basic pseduo code can be found at https://en.wikipedia.org/wiki/Julia_set
    // Note this algorithm is modified
	for (int y = 0; y < height; y++){ // Draws one frame.
		for (int x = 0; x < width; x++){
			zx = 1.5 * (x - width / 2) / (0.5 * zoom * width) + mX;
			zy = (y - height / 2) / (0.5 * zoom * height) + mY;
            int offset = (y * width) + x;
			int iteration = 0;
            float magnitude = 0.0;
            while (iteration < max_iterations){
				ox = zx;
				oy = zy;
				zx = (ox * ox - oy * oy) + cRe;
                zy = (ox * oy + ox * oy) + cIm;
                magnitude = zx * zx + zy * zy;
				if((magnitude) > 4) break;
                iteration++;
			}	

			if(iteration == max_iterations ){// Set color to draw julia
                pixels[y * width * 4 + x * 4 + 0] = (magnitude/2);
                pixels[y * width * 4 + x * 4 + 1] = (magnitude/4);
                pixels[y * width * 4 + x * 4 + 2] = 1.0;
                pixels[y * width * 4 + x * 4 + 3] = 1.0; 
            
			} else {
                pixels[y * width * 4 + x * 4 + 0] = 0;
                pixels[y * width * 4 + x * 4 + 1] = ((iteration/max_iterations)/magnitude);
                pixels[y * width * 4 + x * 4 + 2] = (iteration/max_iterations);
                pixels[y * width * 4 + x * 4 + 3] = 1.0;
            }
		}
	}
}

void animate(){
  
    if (animation == 1){ 
        double old_time = t;
        t = clock();
        double delta_time = (t - old_time)/CLOCKS_PER_SEC;
        double *buf;
        buf = csv_to_array(read_from_file);

        cRe =  (cRe + (.001 * atan(buf[1]))) + 0.001 *  tan(delta_time);
        cIm =  (cIm + (.01  * atan(buf[0]))) + 0.0001 * tan(delta_time);
    }
    display();
}

void display()
{

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear( GL_COLOR_BUFFER_BIT );
    glDrawPixels(width, height, GL_RGBA, GL_FLOAT, pixels);
    
    START_TIMER(julia);
    julia(zoom, mx, my);
    STOP_TIMER(julia);
    print_stats(GET_TIMER(julia));
    
    glutSwapBuffers();
}

void key_listener(unsigned char key, int x, int y){
    switch(key){
        case 'q':
            free(pixels);
            exit(0);
            break;
        case 'w':
            max_iterations += 10;
            break;
        case 's':
            if (max_iterations > 0){
                max_iterations -= 10;
            }
            break;
        //Adjust real
        case 'r':
            cRe += .1;
            break;
        case 'f':
            cRe -= .1;
            break;
        //Adjust Imaginary
        case 'i':
            cIm += .01;
            break;
        case 'k': 
            cIm -= .01;
            break;
        case 32: // Spacebar / Toggle animation
            if (animation == 0){
                animation = 1;
            } else {
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

    // print_stats();

}

void arrow_listener(int key, int x, int y){
    switch(key){
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

    // print_stats();
}

void print_stats(float render_time){
    printf("Time To Render = %f:\n", render_time);
    printf("Iterations %d   real = %f   imaginary = %f   animation = %d\n", max_iterations, cRe, cIm, animation);
    printf("    Camera: pos = %f, %f   zoom = %f\n", mx,my,zoom);
}

double* csv_to_array(char *file){
    static double ret[2];
    FILE *fp;
    fp = fopen(file, "r");
    if (!fp){
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
    while (token != NULL){
        len += strlen(token);
        ret[i] = strtod(token, &token);
        token = strtok(NULL, ",");
        i += 1;
    }

    Step_To_Seek = (len + Step_To_Seek) % 4126;

    return ret;
}