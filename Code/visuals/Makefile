default: clean main par_vis 

main: visualizer.c
	gcc -o vis visualizer.c -lm -lglut -lGLU -lGL

par_vis: par_visualizer.cu
	nvcc -o par_vis par_visualizer.cu -lGL -lm -lglut -lGLU

clean:
	rm -f vis par_vis