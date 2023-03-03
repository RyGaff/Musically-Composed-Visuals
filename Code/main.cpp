#include <GL/glut.h>

GLfloat angle = 0.0f;
GLfloat changer = 1.0f;
void display(void)
{
    glClear(GL_COLOR_BUFFER_BIT);   // Clear the color buffer
    glMatrixMode(GL_MODELVIEW);     // To operate on Model-View matrix
    glLoadIdentity();               // Reset the model-view matrix
    glPushMatrix();                    
    glRotatef(angle, 0.0f, 0.0f, 1.0f); 
    glBegin(GL_QUADS);                  
       glColor3f(  1.0f,  0.0f, 0.0f);     
       glVertex2f(-0.3f, -0.3f);
       glVertex2f( 0.3f, -0.3f);
       glVertex2f( 0.3f,  0.3f);
       glVertex2f(-0.3f,  0.3f);
    glEnd();    
    glPopMatrix();
    glutSwapBuffers();   // Double buffered - swap the front and back buffers

    // Change the rotational angle after each display()
    angle += 1.0f;
}

void timer(int value){
    glutPostRedisplay();
    glutTimerFunc(30, timer, 0);
}

int main(int argc, char** argv)
{
    glutInit(&argc, argv);
    //Init our window
    glutInitDisplayMode(GLUT_SINGLE);
    glutInitWindowSize(400, 300);
    glutInitWindowPosition(100, 100);

    glutCreateWindow("Mandelbrot Set");
    glutDisplayFunc(display);
    glutIdleFunc(display);
    glutTimerFunc(0, timer, 0);
    glutMainLoop();
    return 0;
}