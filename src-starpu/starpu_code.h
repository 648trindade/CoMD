#include <starpu.h>
#include "mytype.h"

// defined in implemented in starpu_code.c
struct params {
    real_t s6, eShift, epsilon, rCut2;
    int nNbrBoxes, nLocalBoxes;
};

static starpu_data_handle_t nAtoms_handle;
static starpu_data_handle_t r_handle;
static starpu_data_handle_t f_handle;
static starpu_data_handle_t U_handle;
static starpu_data_handle_t ePot_handle;
static starpu_data_handle_t nbrBoxes_handle;
static int NTASKS = 1;

static struct starpu_data_filter data_filter = {
    .filter_func = starpu_vector_filter_block,
    .nchildren = 1 // alterado antes da criação das tasks
};

void cpu_func(void *buffers[], void *cl_arg);
#ifdef STARPU_HAVE_CUDA
void gpu_func(void *buffers[], void *cl_arg);
#endif
void ePot_redux_cpu_func(void *descr[], void *cl_arg);
void ePot_init_cpu_func(void *descr[], void *cl_arg);

static struct starpu_codelet ePot_redux_codelet = {
    .cpu_funcs = {ePot_redux_cpu_func},
    .cpu_funcs_name = {"ePot_redux_cpu_func"},
    .modes = {STARPU_RW, STARPU_R},
    .nbuffers = 2,
    .name = "redux"
};

static struct starpu_codelet ePot_init_codelet = {
    .cpu_funcs = {ePot_init_cpu_func},
    .cpu_funcs_name = {"ePot_init_cpu_func"},
    .modes = {STARPU_W},
    .nbuffers = 1,
    .name = "init"
};

// codelet definition
static struct starpu_codelet cl = {
    .cpu_funcs = { cpu_func },
    .cpu_funcs_name = { "cpu_func" },
#ifdef STARPU_USE_CUDA
    .cuda_funcs = { gpu_func },
    .cuda_flags = {STARPU_CUDA_ASYNC},
#endif
    .nbuffers = 6,
    .modes = {STARPU_R, STARPU_R, STARPU_R, STARPU_RW, STARPU_RW, STARPU_REDUX}
    //        boxes,    nAtoms,   atoms->r, atoms->f,  atoms->U,  ePot
};