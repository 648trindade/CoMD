#include <starpu.h>
#include "starpu_code.h"

static __global__ void do_ljForce(
    real_t s6, real_t eShift, real_t epsilon, real_t rCut2, int nNbrBoxes, int nLocalBoxes, int* boxes, int* nAtoms, real3* r, real3* f, real_t* U, real_t* ePot
){
    int slices = (nLocalBoxes + NWORKERS - 1) / NWORKERS;

    for (int iBox = id * slices; iBox < (id + 1) * slices && iBox < nLocalBoxes; iBox++){

      int nIBox = nAtoms[iBox];
      
      // loop over neighbors of iBox
      for (int jTmp=0; jTmp<nNbrBoxes; jTmp++){
            int jBox = boxes[(iBox - id * slices) * nNbrBoxes + jTmp];
            
            //assert(jBox>=0);
            STARPU_ASSERT(jBox>=0);
            
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
}

extern "C" void gpu_func(void *buffers[], void *cl_arg){
    // gathering paramethers
    struct params *params = cl_arg;
    real_t       s6 = params->s6;
    real_t   eShift = params->eShift;
    real_t  epsilon = params->epsilon;
    real_t    rCut2 = params->rCut2;
    int          id = params->id;
    int   nNbrBoxes = params->nNbrBoxes;
    int nLocalBoxes = params->nLocalBoxes;

    //printf("%d\n", starpu_worker_get_id());

    // gathering buffers
    int*   boxes = (   int*) STARPU_VECTOR_GET_PTR(buffers[0]);
    int*  nAtoms = (   int*) STARPU_VECTOR_GET_PTR(buffers[1]);
    real3*     r = ( real3*) STARPU_VECTOR_GET_PTR(buffers[2]);
    real3*     f = ( real3*) STARPU_VECTOR_GET_PTR(buffers[3]);
    real_t*    U = (real_t*) STARPU_VECTOR_GET_PTR(buffers[4]);
    real_t* ePot = (real_t*) STARPU_VARIABLE_GET_PTR(buffers[5]);
    
	do_ljForce<<<1, 1, 0, starpu_cuda_get_local_stream()>>>(s6, eShift, epsilon, rCut2, nNbrBoxes, nLocalBoxes, boxes, nAtoms, r, f, U, ePot);
    cudaError_t cures = cudaStreamSynchronize(starpu_cuda_get_local_stream());
	if (cures)
		STARPU_CUDA_REPORT_ERROR(cures);
}