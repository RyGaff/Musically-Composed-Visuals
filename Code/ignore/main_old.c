#include "dependencies.h"


// Default starting values for cRe and cIm
float const REAL = -0.7;
float const IMAGINARY = 0.27015;
float cRe = REAL;
float cIm = IMAGINARY;

// Dimensions of the window are set in the display function
// This is done soe we can use glutGet
int height;
int width;
int max_iterations = 150;

double zoom = 1;
double mx = 0;
double my = 0;

int animation = 0;
double t = 0.0;

int Step_To_Seek = 0;

//Background color;
float br = 0.0, bg = 0.0, bb = 0.0;

void display();
void animate();
void julia(double zoom, double mX, double mY);
void key_listener(unsigned char key, int x, int y);
void arrow_listener(int key, int x, int y);
double random_interval() {return (double)rand()/(double)RAND_MAX;}
void print_stats();
double* csv_to_array(char* file); 

int main( int argc, char** argv )
{
    csv_to_array("cat.csv");

    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize( 1000, 1000);
    glutInitWindowPosition(950,0);
    glutCreateWindow( "Julia" );

    glutDisplayFunc(animate);
    glutIdleFunc(animate);
    glutKeyboardFunc(key_listener);
    glutSpecialFunc(arrow_listener);
    glutMainLoop();
    return 0;
}

void julia(double zoom, double mX, double mY)
{
    
	double zx, zy, ox, oy;

    glBegin( GL_POINTS ); // start drawing in single pixel mode, 
    // KILLS performance but should be easily parallizable with cuda I hope
    // If failure to parallelize with cuda with draw with triangles and textures instead

    // algorithm to draw the julia set. 
    // basic pseduo code can be found at https://en.wikipedia.org/wiki/Julia_set
    // Note this algorithm is modified
	for (int y = 0; y < height; y++){ // Draws one frame.
		for (int x = 0; x < width; x++){
			zx = 1.5 * (x - width / 2) / (0.5 * zoom * width) + mX;
			zy = (y - height / 2) / (0.5 * zoom * height) + mY;

			int iteration = 0;
			for (iteration; iteration < max_iterations; iteration++){
				ox = zx;
				oy = zy;
				zx = (ox * ox - oy * oy) + cRe;
				// zy = (2 * ox * oy + cIm);
                zy = (ox * oy + ox * oy) + cIm;
				if((zx * zx + zy * zy) > 4) break;
			}	
				if(iteration == max_iterations ){// Set color to draw julia
                glColor3f( oy, zx - zy, oy-ox); 
                // glColor3f(ox + oy, oy, abs(ox-oy)); // Set the color of everything not part of the julia set
                glVertex2i( x, y );
			}
            else { // Set color to draw pixels not apart of julia
                glColor3f(br,bg,bb);
                // glColor3f(abs(oy -ox), ox, ox + oy);
                // glColor3f(abs(ox-oy) + 0.1,0.0,abs(ox-oy) + .01);
                glVertex2i( x, y );
            }
		}
	}
    glEnd();
}

void animate(){
  
    if (animation == 1){ 
        
        double *buf;
        buf = csv_to_array("cat.csv");

        double old_time = t;
        t = clock();
        double delta_time = (t - old_time);

        cRe = (cRe + buf[0]/1000000) + 0.005 * sin(delta_time/zoom);
        cIm = (cIm + buf[1]/10000000) + 0.005 * cos(delta_time/zoom); 

        // printf("test %f  %f\n", cRe, cIm);
        br = .01 * remainder(buf[0],Step_To_Seek) / 255;
        bg = .01 * remainder(buf[1],Step_To_Seek) / 255;
        bb = .01 * remainder(cRe, cIm)/255;

    }
    display();
}

void display()
{
    height = glutGet( GLUT_WINDOW_HEIGHT );
    width = glutGet( GLUT_WINDOW_WIDTH );

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear( GL_COLOR_BUFFER_BIT );
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glOrtho( 0, width, 0, height, -1, 1 );
    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

    julia(zoom, mx, my);
    glutSwapBuffers();
}

void key_listener(unsigned char key, int x, int y){
    switch(key){
        case 'q':
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
    print_stats();

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

    print_stats();
}

void print_stats(){
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