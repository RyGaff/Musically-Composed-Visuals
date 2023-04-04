// #define GLM_FORCE_CUDA
// #include <glm/glm.hpp>
#include <math.h>
#include <stdio.h>
#include <GL/glut.h>

float cRe = -0.7;
float cIm = 0.27015;

// Dimensions of the window are set in the display function
// This is done soe we can use glutGet
int height;
int width;
int MAX_ITERATIONS = 500;

void julia(double zoom, double mX, double mY)
{
    
	double zx, zy, ox, oy;

    glBegin( GL_POINTS ); // start drawing in single pixel mode

    // algorithm to draw the julia set. 
    // basic pseduo code can be found at https://en.wikipedia.org/wiki/Julia_set
	for (int y = 0; y < height; y++){
		for (int x = 0; x < width; x++){
			zx = 1.5 * (x - width / 2) / (0.5 * zoom * width) + mX;
			zy = (y - height / 2) / (0.5 * zoom * height) + mY;

			int iteration = 0;
			for (iteration; iteration < MAX_ITERATIONS; iteration++){
				ox = zx;
				oy = zy;
				zx = ox * ox - oy * oy + cRe;
				zy = 2 * ox * oy + cIm;

				if((zx * zx + zy * zy) > 4) break;
			}	
				if(iteration == MAX_ITERATIONS ){
                glColor3f( 1.0, 1.0, 1.0 ); // Set color to draw julia
                glVertex2i( x, y );
			}
            else {
                glColor3f( 0.0, 0.0, 0.0 ); // Set the color of everything not part of the julia set
                glVertex2i( x, y );
            }
		}
	}
    glEnd();
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

    //TODO Allow the (zoom, mx, my) parameters to be changed via user input or some other method
    julia( 1.0, 0.0, 0.0);
    glutSwapBuffers();
}

void key_listener(unsigned char key, int x, int y){
    switch(key){
        case 'q':
            exit(0);
            break;
        case 'w':
            MAX_ITERATIONS += 10;
            break;
        case 's':
            if (MAX_ITERATIONS > 0){
                MAX_ITERATIONS -= 10;
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
            cIm += .1;
            break;
        case 'k':
            cIm -= .1;
            break;
    }
    printf("Iterations %d   real = %f    imaginary = %f\n", MAX_ITERATIONS, cRe, cIm);

}

int main( int argc, char** argv )
{
    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
    glutInitWindowSize( 1000, 1000);
    glutCreateWindow( "Julia" );

    glutDisplayFunc( display );
    glutIdleFunc(display);
    glutKeyboardFunc(key_listener);
    glutMainLoop();
    return 0;
}