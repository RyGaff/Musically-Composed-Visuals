# Video Links
Draft: https://youtu.be/gugts8mUX-U
Final: https://youtu.be/POcOWgQ5GfU

#Snipits from the final writeup
# Team

Justin Choi, Ryan Gaffney, Ian Lips, Matt Dim

# Goal

Musically-Composed-Visuals Parallelizing audio transformation into
different classifications regarding image generation and using graphics
to display our results (in the form of the julia set fractal). There are already ways of breaking music into components
through fourier. These components would split songs into components such
as basslines, vocals, accompaniment, etc.


# Overview
  The first aspect of our project is our audio processing. We perform FFT and DFT on a wav file in order to compress the information from the audio into a csv format. We then use that csv as an input into our visualizer to generate images throughout the audio. Our csv data consists of the frame by frame frequencies of the audio and stores its instances (Figure CSV GRAPH). We then generate images using OpenGL, utilizing the julia set to create fractals. With the frame by frame frequencies stored in our csv, we create and cycle through a multitude of generated images to create an animated julia set, where each frame corresponds to its audio counterpart. Our goal with this project was to parallelize FFT, DFT, and the Julia fractal. We wanted to ensure speed up while maintaining 100% accuracy on FFT, and DFT. To parallelize these we had to actually change how some of the math worked because of limitations with cuda, which we will build upon later in the writeup. On the julia fractal our focus was all about speed, while still keeping the Julia fractal intact of course. Since the julia fractal was just being used to visualize the FFT or DFT we didn’t entirely care if something was slightly off. As a result the whole focus there was speed. 

  ---------------------------- ---------------------------- ---------------------------- -- -- --
# Polynomial Julia Set images
  ![image](/Latex/imgs/image1.png)    ![image](/Latex/imgs/image2.png)   ![image](/Latex/imgs/image3.png)        
  ---------------------------- ---------------------------- ---------------------------- -- -- --

# Components of our Project

FFT (Fast Fourier Transform) is a mathematical algorithm that efficiently transforms time-domain signals into an equivalent frequency-domain representation. In simpler terms, it analyzes the different frequencies of a signal. When a signal is converted into frequencies, we can identify specific components we can easily modify. The main idea is to break down these components into smaller forms of themselves so that they can be processed simply and efficiently. The main (serial) way of doing this is using recursion to divide the signal into smaller chunks until we find the smallest component, which results in a frequency-domain representation that we use for our visualizations. For parallel purposes, we broke the recursive algorithm into an iterative version. The iterative version functions the same and is used for timing comparison and speedup. An important distinction to make with the FFT algorithm is that it is O(nlog(n) but requires more memory overhead when compared to the Discrete Fourier Transform algorithm.

Similar to the FFT algorithm, the DFT (Discrete Fourier Transform) decomposes a signal into basic frequency components. The way DFT works is by operating on a finite set of discrete samples of the signal and produces a discrete set of frequencies that match the original signal. One of the main differences from FFT is that DFT is more computationally intensive, being O(n^2) but has less memory overhead.

The Julia Set is a fractal set that associates with complex quadratic polynomial functions. It creates a visual that color codes points in a plane based on if they are in the set or not. Any point outside of the set is typically assigned a simple single color, while the points that escape to infinity are various shades that are dependent on the ratio of iterations to our iteration cap, and the magnitude, which is the value used to break out of the loop. This results in a fractal with various shaded colors and satisfying shapes. Due to the nature of the algorithm, the behavior of the Julia set is described as “chaotic”.

The generation of Julia set fractals starts with the initialization of a complex number z = x+yi where i^2 = -1 and x and y are coordinates in the range from -2 to 2. Z is then updated on loop using z = z^2 + c where c is another complex number that gives a 

::: thebibliography
9 N. Corporation, "OpenGL interoperability," *NVIDIA Documentation Hub*,
2023. \[Online\]. Available:
<https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OPENGL.html>.
\[Accessed: 03-Mar-2023\].

K. Kolsha, "Kolsha/STM32-audio-visualizer: Audio visualizer based on
stm32 and Extension Board.," GitHub, 2018. \[Online\]. Available:
<https://github.com/Kolsha/STM32-AUDIO-VISUALIZER>. \[Accessed:
03-Mar-2023\].

C. Williams, "Online fractal generator," UsefulJS, 2014. \[Online\].
Available: <http://usefuljs.net/fractals/>. \[Accessed: 03-Mar-2023\].

C. Zelga, A. Ulug, and J. Gilbert, "ASENAULUG/music-visualizer: A music
visualizer which displays the real time fast Fourier transform (FFT) of
music on an RGB led matrix," GitHub, 2019. \[Online\]. Available:
<https://github.com/asenaulug/music-visualizer>. \[Accessed:
03-Mar-2023\].
:::
