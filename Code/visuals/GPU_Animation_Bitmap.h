#include "gl_helper.h"
#include "cuda.h"
#include "cuda_gl_interop.h"
#include <stdio.h>
#include <string.h>


PFNGLBINDBUFFERARBPROC    glBindBuffer     = NULL;
PFNGLDELETEBUFFERSARBPROC glDeleteBuffers  = NULL;
PFNGLGENBUFFERSARBPROC    glGenBuffers     = NULL;
PFNGLBUFFERDATAARBPROC    glBufferData     = NULL;

struct GPUAnimBitmap {
    GLuint  bufferObj;
    cudaGraphicsResource *resource;
    int     width, height;
    void    *dataBlock;
    void (*fAnim)(uchar4*,void*);
    void (*animExit)(void*);
    void (*clickDrag)(void*,int,int,int,int);
    int     dragStartX, dragStartY;

    GPUAnimBitmap( int w, int h, void *d = NULL ) {
        width = w;
        height = h;
        dataBlock = d;
        clickDrag = NULL;

        // first, find a CUDA device and set it to graphic interop
        cudaDeviceProp  prop;
        int dev;
        memset( &prop, 0, sizeof( cudaDeviceProp ) );
        prop.major = 1;
        prop.minor = 0;
        cudaChooseDevice( &dev, &prop );

        // a bug in the Windows GLUT implementation prevents us from
        // passing zero arguments to glutInit()
        int c=1;
        char* dummy = (char*)alloca(1);
        glutInit( &c, &dummy );
        glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
        glutInitWindowSize( width, height );
        glutInitWindowPosition(950,0);
        glutCreateWindow( "Julia" );

        glBindBuffer    = (PFNGLBINDBUFFERARBPROC)GET_PROC_ADDRESS("glBindBuffer");
        glDeleteBuffers = (PFNGLDELETEBUFFERSARBPROC)GET_PROC_ADDRESS("glDeleteBuffers");
        glGenBuffers    = (PFNGLGENBUFFERSARBPROC)GET_PROC_ADDRESS("glGenBuffers");
        glBufferData    = (PFNGLBUFFERDATAARBPROC)GET_PROC_ADDRESS("glBufferData");

        glGenBuffers( 1, &bufferObj );
        glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj );
        glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, width * height * 4, NULL, GL_DYNAMIC_DRAW_ARB );

        cudaGraphicsGLRegisterBuffer( &resource, bufferObj, cudaGraphicsMapFlagsNone );
    }

    ~GPUAnimBitmap() {
        free_resources();
    }

    void free_resources( void ) {
        cudaGraphicsUnregisterResource( resource );

        glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
        glDeleteBuffers( 1, &bufferObj );
        fcloseall(); 
    }

    void anim_and_exit( void (*f)(uchar4*,void*), void(*e)(void*) ) {
        GPUAnimBitmap**   bitmap = get_bitmap_ptr();
        *bitmap = this;
        fAnim = f;
        animExit = e;
        glutDisplayFunc( Draw );
        glutIdleFunc( idle_func );
    }

    // static method used for glut callbacks
    static GPUAnimBitmap** get_bitmap_ptr( void ) {
        static GPUAnimBitmap*   gBitmap;
        return &gBitmap;
    }

    // static method used for glut callbacks
    static void idle_func( void ) {
        GPUAnimBitmap* bitmap = *(get_bitmap_ptr());
        uchar4* devPtr;
        size_t size;

        cudaGraphicsMapResources( 1, &(bitmap->resource), NULL );
        cudaGraphicsResourceGetMappedPointer( (void**)&devPtr, &size, bitmap->resource);

        bitmap->fAnim( devPtr, bitmap->dataBlock);

        cudaGraphicsUnmapResources( 1, &(bitmap->resource), NULL );

        glutPostRedisplay();
    }

    // static method used for glut callbacks
    static void Draw( void ) {
        GPUAnimBitmap*   bitmap = *(get_bitmap_ptr());
        glClearColor( 0.0, 0.0, 0.0, 1.0 );
        glClear( GL_COLOR_BUFFER_BIT );
        glDrawPixels( bitmap->width, bitmap->height, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
        glutSwapBuffers();
    }
};
