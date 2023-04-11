default: main par_vis

main: main.c
	gcc -o main main.c -lm -lglut -lGLU -lGL

par_vis: par_vis.cu
	nvcc -o par_vis par_vis.cu -lGL -lm -lglut -lGLU

clean:
	rm -f par_vis serial_vis main