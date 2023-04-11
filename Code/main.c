#include "dependencies.h"
#include "gl_helper.h"
#include "cuda_gl_interop.h"



// Default starting values for cRe and cIm
float const REAL = -0.7;
float const IMAGINARY = 0.27015;
float cRe = REAL;
float cIm = IMAGINARY;

// Dimensions of the window are set in the display function
// This is done soe we can use glutGet
int const height = 1024;
int const width = 1024;
int max_iterations = 150;

double zoom = 1;
double mx = 0;
double my = 0;

int animation = 0;
double t = 0.0;

int Step_To_Seek = 0;
unsigned int frames = 0;

//Background color;
float br = 0.0, bg = 0.0, bb = 0.0;
// GLubyte* pixels;
// float pixels[1024][1024][3];
// uchar4 pixels[width * height * 4];
// uchar4 *pixels;

float *pixels;
// unsigned char *pixels;
// uchar4 *pixels;

// float ***pixels;

void display();
void animate();
void julia(double zoom, double mX, double mY);
void key_listener(unsigned char key, int x, int y);
void arrow_listener(int key, int x, int y);
void print_stats(float render_time);
double* csv_to_array(char* file); 

int main( int argc, char** argv )
{
    // csv_to_array("cat.csv");
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

    // glBegin( GL_POINTS ); // start drawing in single pixel mode, 
    // KILLS performance but should be easily parallizable with cuda I hope
    // If failure to parallelize with cuda with draw with triangles and textures instead

    // algorithm to draw the julia set. 
    // basic pseduo code can be found at https://en.wikipedia.org/wiki/Julia_set
    // Note this algorithm is modified
	for (int y = 0; y < height; y++){ // Draws one frame.
        // unsigned long long offset = y * 1024;
		for (int x = 0; x < width; x++){
            // printf("offset = %d\n", offset);
			zx = 1.5 * (x - width / 2) / (0.5 * zoom * width) + mX;
			zy = (y - height / 2) / (0.5 * zoom * height) + mY;
            // int offset = x + y * 64 * 16;
            int offset = (y * width) + x;
			int iteration = 0;
			// for (iteration; iteration < max_iterations; iteration++){
            while (iteration < max_iterations){
				ox = zx;
				oy = zy;
				zx = (ox * ox - oy * oy) + cRe;
                zy = (ox * oy + ox * oy) + cIm;
				if((zx * zx + zy * zy) > 4){
                    pixels[y * width * 4 + x * 4 + 0] = 0.0;
                    pixels[y * width * 4 + x * 4 + 1] = 0.0;
                    pixels[y * width * 4 + x * 4 + 2] = 0.0;
                    pixels[y * width * 4 + x * 4 + 3] = 1.0;

                    // pixels[offset].x = 0;
                    // pixels[offset].y = 0;
                    // pixels[offset].z = 0;
                    // pixels[offset].w = 255;
                    break;
                }
                iteration++;
			}	
            // offset++;

			if(iteration == max_iterations ){// Set color to draw julia
                pixels[y * width * 4 + x * 4 + 0] = 0;
                pixels[y * width * 4 + x * 4 + 1] = ox*ox;
                pixels[y * width * 4 + x * 4 + 2] = oy;
                pixels[y * width * 4 + x * 4 + 3] = 1.0;

                // pixels[offset].x = (unsigned char) 255 * oy;
                // pixels[offset].y = (unsigned char) 255 * (zx-zy);
                // pixels[offset].z = (unsigned char) 255 * (oy-ox);
                // pixels[offset].w = 255 * 1.0;

                // pixels[offset].x = (unsigned char) 255 * 1;
                // pixels[offset].y = (unsigned char) 255 * oy;
                // pixels[offset].z = (unsigned char) 255 * ox;
                // pixels[offset].w = 255 * 1.0;
            
			}
		}
	}
}

void animate(){
  
    if (animation == 1){ 
        
        double *buf;
        buf = csv_to_array("cat.csv");

        double old_time = t;
        t = clock();
        double delta_time = (t - old_time);

        cRe = (cRe + buf[0]/10000000)  + 0.005 * sin(delta_time/zoom);
        cIm = (cIm + buf[1]/100000000) + 0.005 * cos(delta_time/zoom); 

        // br = .01 * remainder(buf[0],step_To_Seek) / 255;
        // bg = .01 * remainder(buf[1],step_To_Seek) / 255;
        // bb = .01 * remainder(cRe, cIm)/255;
        // print_stats();

        // cRe = (cRe ) + 0.0001 * sin(delta_time + 10);
        // cIm = (cIm ) + 0.0001 * cos(delta_time + 10); 
        // printf("test %f  %f\n", cRe, cIm);
        // br = .01 * remainder(buf[0],step_To_Seek) / 255;
        // bg = .01 * remainder(buf[1],step_To_Seek) / 255;
        // bb = .01 * remainder(cRe, cIm)/255;

    }
    display();
}

void display()
{
    // height = glutGet( GLUT_WINDOW_HEIGHT );
    // width = glutGet( GLUT_WINDOW_WIDTH );

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear( GL_COLOR_BUFFER_BIT );
    // glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    glDrawPixels(width, height, GL_RGBA, GL_FLOAT, pixels);
    // glDrawPixels(width, height, GL_RGBA, GL_DOUBLE, pixels);
    START_TIMER(julia);
    julia(zoom, mx, my);
    STOP_TIMER(julia);
    print_stats(GET_TIMER(julia));
    
    // frames++;

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

    // printf("Iterations %d", max_iterations);
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
        // printf("Token = %s", token);
        len += strlen(token);
        ret[i] = strtod(token, &token);
        token = strtok(NULL, ",");
        i += 1;
    }

    Step_To_Seek = (len + Step_To_Seek) % 4126;

    return ret;
}