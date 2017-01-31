#include <starpu.h>

#include "mytype.h"

starpu_data_handle_t* create_and_register_vector_handle(void* data, int NX, int elem_size);

starpu_data_handle_t* create_and_register_matrix_handle(void* data, int width, int height, int size);

starpu_data_handle_t* create_and_register_variable_handle(void* var, int size);

void unregister_and_destroy_data_handles(starpu_data_handle_t* handles[], int num_handles);

// defined in implemented in starpu_code.c
struct params {
    real_t s6, eShift, epsilon, rCut2;
    int iBox, nNbrBoxes;
};

void cpu_func(void *buffers[], void *cl_arg);
void ePot_redux_cpu_func(void *descr[], void *cl_arg);
void ePot_init_cpu_func(void *descr[], void *cl_arg);