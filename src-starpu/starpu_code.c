#include "starpu_code.h"
#include "memUtils.h"
#include "linkCells.h"
#include <string.h>

// create and register a handle for a vector
data_handle* create_and_register_vector_handle(void* data, int NX, int elem_size){
    data_handle *handle = comdMalloc(sizeof(data_handle));
    memset(handle, 0, sizeof(data_handle));
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
data_handle* create_and_register_variable_handle(void* var, int size){
    data_handle *handle = comdMalloc(sizeof(data_handle));
    memset(handle, 0, sizeof(data_handle));
    starpu_variable_data_register(
        handle,
        STARPU_MAIN_RAM,
        (uintptr_t)var,
        size
    );
    return handle;
}

// create and start a task
void create_and_start_task(data_handle* handles[], void* params, int params_size){

    // codelet definition
    struct starpu_codelet cl = {
        .cpu_funcs = { cpu_func },
        .cpu_funcs_name = { "cpu_func" },
        .nbuffers = 6,
        .modes = {STARPU_R, STARPU_R, STARPU_R, STARPU_RW, STARPU_RW, STARPU_RW}
        //        jBoxes,   nAtoms,   atoms->r, atoms->f,  atoms->U,  ePot
    };

    // task definition
    struct starpu_task *task = starpu_task_create();
    task->cl = &cl;
    task->cl_arg = params;
    task->cl_arg_size = params_size;
    // FIXME: only works with synchrony (SEGFAULT for asynchronous)
    task->synchronous = 1;

    // assign handles to task
    int i;
    for(int i=0; i<6; i++)
        task->handles[i] = *(handles[i]);
    
    // submit task
    int ret = starpu_task_submit(task);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
}

// ljForce main loop (iterate over all boxes)
void cpu_func(void *buffers[], void *cl_arg){
    
    // gathering paramethers
    struct params *params = cl_arg;
    real_t      s6 = params->s6;
    real_t  eShift = params->eShift;
    real_t epsilon = params->epsilon;
    real_t   rCut2 = params->rCut2;
    int       iBox = params->iBox;
    int  nNbrBoxes = params->nNbrBoxes;

    // gathering buffers
    int*  jBoxes = (   int*) STARPU_VECTOR_GET_PTR(buffers[0]);
    int*  nAtoms = (   int*) STARPU_VECTOR_GET_PTR(buffers[1]);
    real3*     r = ( real3*) STARPU_VECTOR_GET_PTR(buffers[2]);
    real3*     f = ( real3*) STARPU_VECTOR_GET_PTR(buffers[3]);
    real_t*    U = (real_t*) STARPU_VECTOR_GET_PTR(buffers[4]);
    real_t* ePot = (real_t*) STARPU_VECTOR_GET_PTR(buffers[5]);

    int nIBox = nAtoms[iBox];
   
    // loop over neighbors of iBox
    for (int jTmp=0; jTmp<nNbrBoxes; jTmp++){
         int jBox = jBoxes[jTmp];
         
         assert(jBox>=0);
         
         int nJBox = nAtoms[jBox];
         
         // loop over atoms in iBox
         for (int iOff=MAXATOMS*iBox; iOff<(iBox*MAXATOMS+nIBox); iOff++){

            // loop over atoms in jBox
            for (int jOff=jBox*MAXATOMS; jOff<(jBox*MAXATOMS+nJBox); jOff++){
                real3 dr;
                real_t r2 = 0.0;
                for (int m=0; m<3; m++){
                    dr[m] = r[iOff][m]-r[jOff][m];
                    r2+=dr[m]*dr[m];
                }

                if ( r2 <= rCut2 && r2 > 0.0){

                    // Important note:
                    // from this point on r actually refers to 1.0/r
                    r2 = 1.0/r2;
                    real_t r6 = s6 * (r2*r2*r2);
                    real_t eLocal = r6 * (r6 - 1.0) - eShift;
                    U[iOff] += 0.5*eLocal;
                    // zona critica
                    *ePot += 0.5*eLocal;

                    // different formulation to avoid sqrt computation
                    real_t fr = - 4.0*epsilon*r6*r2*(12.0*r6 - 6.0);
                    for (int m=0; m<3; m++)
                        f[iOff][m] -= dr[m]*fr;
                }
            } // loop over atoms in jBox
        } // loop over atoms in iBox
    } // loop over neighbor boxes
}

// wait for tasks and unregister/free handles
void unregister_and_destroy_data_handles(data_handle* handles[], int num_handles){
    int i;
    starpu_task_wait_for_all();
    for (i = 0; i < num_handles; i++){
        starpu_data_unregister(*(handles[i]));
        comdFree(handles[i]);
    }
}