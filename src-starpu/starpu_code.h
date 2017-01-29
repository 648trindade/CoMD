#include <starpu.h>

#include "mytype.h"

typedef starpu_data_handle_t data_handle;

struct params {
    real_t s6, eShift, *ePot, epsilon, rCut2;
    int iBox, nNbrBoxes;
};

data_handle* create_and_registar_data_handle(void* data, int NX, int elem_size);

void create_and_start_task(data_handle* handles[], int num_handles, void* params, int params_size);

void cpu_func(void *buffers[], void *cl_arg);