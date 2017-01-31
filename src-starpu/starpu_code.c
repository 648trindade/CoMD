#include "starpu_code.h"
#include "memUtils.h"
#include "linkCells.h"
#include <string.h>

// create and register a handle for a vector
starpu_data_handle_t* create_and_register_vector_handle(void* data, int NX, int elem_size){
    starpu_data_handle_t *handle = comdMalloc(sizeof(starpu_data_handle_t));
    memset(handle, 0, sizeof(starpu_data_handle_t));
    starpu_vector_data_register(
        handle,
        STARPU_MAIN_RAM,
        (uintptr_t)data,
        NX,
        elem_size
    );
    return handle;
}

// create and register a handle for a variable
starpu_data_handle_t* create_and_register_matrix_handle(void* var, int width, int height, int size){
    starpu_data_handle_t *handle = comdMalloc(sizeof(starpu_data_handle_t));
    memset(handle, 0, sizeof(starpu_data_handle_t));
    starpu_matrix_data_register(
        handle,
        STARPU_MAIN_RAM,
        (uintptr_t)var,
        width,
        width,
        height,
        size
    );
    return handle;
}

// create and register a handle for a variable
starpu_data_handle_t* create_and_register_variable_handle(void* var, int size){
    starpu_data_handle_t *handle = comdMalloc(sizeof(starpu_data_handle_t));
    memset(handle, 0, sizeof(starpu_data_handle_t));
    starpu_variable_data_register(
        handle,
        STARPU_MAIN_RAM,
        (uintptr_t)var,
        size
    );
    return handle;
}

// unregister/free handles
void unregister_and_destroy_data_handles(starpu_data_handle_t* handles[], int num_handles){
    int i;
    for (i = 0; i < num_handles; i++){
        starpu_data_unregister(*(handles[i]));
        comdFree(handles[i]);
    }
}