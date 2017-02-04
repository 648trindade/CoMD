#include <starpu.h>
#include "mytype.h"

static int NWORKERS = 1;

// defined in implemented in starpu_code.c
struct params {
    real_t s6, eShift, epsilon, rCut2;
    int id, nNbrBoxes, nLocalBoxes;
};

void cpu_func(void *buffers[], void *cl_arg);
#ifdef STARPU_HAVE_CUDA
void gpu_func(void *buffers[], void *cl_arg);
#endif
void ePot_redux_cpu_func(void *descr[], void *cl_arg);
void ePot_init_cpu_func(void *descr[], void *cl_arg);