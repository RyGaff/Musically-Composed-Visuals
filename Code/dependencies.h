#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include "gl_helper.h"

struct timeval tv;

// timing macros (must first declare "struct timeval tv")
#define START_TIMER(NAME) gettimeofday(&tv, NULL); \
    double NAME ## _time = tv.tv_sec+(tv.tv_usec/1000000.0);
#define STOP_TIMER(NAME) gettimeofday(&tv, NULL); \
    NAME ## _time = tv.tv_sec+(tv.tv_usec/1000000.0) - (NAME ## _time);
#define GET_TIMER(NAME) (NAME##_time)