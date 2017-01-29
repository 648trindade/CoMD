#include "starpu_code.h"
#include "memUtils.h"
#include "linkCells.h"

data_handle* create_and_registar_data_handle(void* data, int NX, int elem_size){
    data_handle *handle = comdMalloc(sizeof(data_handle));
    starpu_vector_data_register(
        handle,
        STARPU_MAIN_RAM,
        (uintptr_t)data,
        NX,
        elem_size
    );
    return handle;
}

void create_and_start_task(data_handle* handles[], int num_handles, void* params, int params_size){

    struct starpu_codelet cl = {
        .cpu_funcs = { cpu_func },
        .cpu_funcs_name = { "cpu_func" },
        .nbuffers = num_handles,
        .modes = { STARPU_RW }
    };

    struct starpu_task *task = starpu_task_create();
    task->cl = &cl; /* Pointer to the codelet defined above */
    task->cl_arg = params;
    task->cl_arg_size = params_size;
    task->synchronous = 1;

    int i;
    for(int i=0; i<num_handles; i++)
        task->handles[i] = *(handles[i]);

    int ret = starpu_task_submit(task);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
    
    starpu_data_unregister(*(handles[0]));
}

void cpu_func(void *buffers[], void *cl_arg){

    struct params *params = cl_arg;
    real_t      s6 = params->s6;
    real_t  eShift = params->eShift;
    real_t   *ePot = params->ePot;
    real_t epsilon = params->epsilon;
    real_t   rCut2 = params->rCut2;
    int       iBox = params->iBox;
    int  nNbrBoxes = params->nNbrBoxes;

    int *jBoxes = (   int*) STARPU_VECTOR_GET_PTR(buffers[0]);
    int *nAtoms = (   int*) STARPU_VECTOR_GET_PTR(buffers[1]);
    real3*    r = ( real3*) STARPU_VECTOR_GET_PTR(buffers[2]);
    real3*    f = ( real3*) STARPU_VECTOR_GET_PTR(buffers[3]);
    real_t*   U = (real_t*) STARPU_VECTOR_GET_PTR(buffers[4]);

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