#include <starpu.h>

#include "mytype.h"

#define data_handle starpu_data_handle_t

struct params {
    real_t s6, eShift, epsilon, rCut2;
    int iBox, nNbrBoxes;
};

data_handle* create_and_register_vector_handle(void* data, int NX, int elem_size);

data_handle* create_and_register_variable_handle(void* var, int size);

void create_and_start_task(data_handle* handles[], void* params, int params_size);

void cpu_func(void *buffers[], void *cl_arg);

void unregister_and_destroy_data_handles(data_handle* handles[], int num_handles);