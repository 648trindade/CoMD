#include <starpu.h>
#include "mytype.h"

#define MAXATOMS 64
#define MAXTHREADS 64

static __global__ void do_ljForce(
    real_t s6, real_t eShift, real_t epsilon, real_t rCut2, int nNbrBoxes,
    int nLocalBoxes, int* nbrBoxes, int* nAtoms, real3* r, real3* f, real_t* U,
    real_t* ePot, size_t nbrBoxes_offset, size_t nbrBoxes_nx,
    size_t iOff_offset
){
    extern __shared__ real_t ePot_data[];
    unsigned int tid = threadIdx.x;
    
    ePot_data[tid] = 0.0;

    //*ePot = 0.0;
    int iBox = nbrBoxes_offset + tid;
    //for (int iBox = nbrBoxes_offset; iBox < nbrBoxes_offset + nbrBoxes_nx && iBox < nLocalBoxes; iBox++){

      int nIBox = nAtoms[iBox];
      
      // loop over neighbors of iBox
      for (int jTmp=0; jTmp<nNbrBoxes; jTmp++){
            int jBox = nbrBoxes[(iBox - nbrBoxes_offset) * nNbrBoxes + jTmp];
            
            assert(jBox>=0);
            
            int nJBox = nAtoms[jBox];
            
            // loop over atoms in iBox
            for (int iOff=MAXATOMS*iBox; iOff<(iBox*MAXATOMS+nIBox); iOff++){
               // iOff traduzido para uso em dados particionados
               int task_iOff = iOff - iOff_offset;

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
                     U[task_iOff] += 0.5*eLocal;
                     //*ePot = *ePot + 0.5*eLocal;
                     ePot_data[tid] += 0.5*eLocal;

                     // different formulation to avoid sqrt computation
                     real_t fr = - 4.0*epsilon*r6*r2*(12.0*r6 - 6.0);
                     for (int m=0; m<3; m++)
                           f[task_iOff][m] -= dr[m]*fr;
                  }
               } // loop over atoms in jBox
         } // loop over atoms in iBox
      } // loop over neighbor boxes
   // }
    __syncthreads();
    // do reduction in shared mem
    for (unsigned int s=1; s < blockDim.x; s *= 2)
        if ((tid % (2*s) == 0) && (tid + s < blockDim.x))
            ePot_data[tid] += ePot_data[tid + s];
        __syncthreads();
    if (tid == 0)
        *ePot += ePot_data[0];
    return;
}

extern "C" void cuda_func(void *buffers[], void *cl_arg){
    // Angariando parâmetros
    real_t s6, eShift, epsilon, rCut2;
    int    nNbrBoxes, nLocalBoxes, id;
    
    starpu_codelet_unpack_args(cl_arg, &s6, &eShift, &epsilon, &rCut2, &nNbrBoxes, &nLocalBoxes, &id);
    
    // Angariando buffers
    int* nbrBoxes = (   int*) STARPU_VECTOR_GET_PTR(buffers[0]);
    int*   nAtoms = (   int*) STARPU_VECTOR_GET_PTR(buffers[1]);
    real3*      r = ( real3*) STARPU_VECTOR_GET_PTR(buffers[2]);
    real3*      f = ( real3*) STARPU_VECTOR_GET_PTR(buffers[3]);
    real_t*     U = (real_t*) STARPU_VECTOR_GET_PTR(buffers[4]);
    real_t*  ePot = (real_t*) STARPU_VARIABLE_GET_PTR(buffers[5]);
    
    // Angariando offsets e números de elementos
    //size_t nbrBoxes_offset = (size_t) STARPU_VECTOR_GET_OFFSET(buffers[0]);
    size_t nbrBoxes_nx = (size_t) STARPU_VECTOR_GET_NX(buffers[0]);
    //size_t iOff_offset = (size_t) STARPU_VECTOR_GET_OFFSET(buffers[4]);
    size_t f_nx = (size_t) STARPU_VECTOR_GET_NX(buffers[3]);
    size_t U_nx = (size_t) STARPU_VECTOR_GET_NX(buffers[4]);

    // Conferindo se offsets e tamanhos estão dentro do esperado
    //STARPU_ASSERT((nbrBoxes_offset / sizeof(int)) % nNbrBoxes == 0);
    STARPU_ASSERT(nbrBoxes_nx % nNbrBoxes == 0);
    //STARPU_ASSERT((iOff_offset / sizeof(real_t)) % MAXATOMS == 0);
    STARPU_ASSERT(U_nx % MAXATOMS == 0);
    STARPU_ASSERT(f_nx % MAXATOMS == 0);
    
    // Calculando offsets e tamanhos reais
    //nbrBoxes_offset /= nNbrBoxes * sizeof(int);
    size_t nbrBoxes_offset = (nbrBoxes_nx/nNbrBoxes) * id;
    nbrBoxes_nx     /= nNbrBoxes;
    size_t iOff_offset = U_nx * id;
    //iOff_offset     /= sizeof(real_t);

    // int n_threads = STARPU_MIN(MAXTHREADS, nbrBoxes_nx);
    // int loops_per_thread = (nbrBoxes_nx + n_threads - 1) / n_threads;
    
	do_ljForce<<<1, nbrBoxes_nx, nbrBoxes_nx * sizeof(real_t), starpu_cuda_get_local_stream()>>>(s6, eShift, epsilon, rCut2, nNbrBoxes, nLocalBoxes, nbrBoxes, nAtoms, r, f, U, ePot, nbrBoxes_offset, nbrBoxes_nx, iOff_offset);
    cudaError_t cures = cudaStreamSynchronize(starpu_cuda_get_local_stream());
	if (cures)
		STARPU_CUDA_REPORT_ERROR(cures);
}

static __global__ void cuda_redux(real_t *ePot, real_t *ePot_worker){
    *ePot = *ePot + *ePot_worker;
    return;
}

extern "C" void ePot_redux_cuda_func(void *descr[], void *cl_arg){
    real_t *ePot = (real_t *)STARPU_VARIABLE_GET_PTR(descr[0]);
	real_t *ePot_worker = (real_t *)STARPU_VARIABLE_GET_PTR(descr[1]);

	cuda_redux<<<1,1, 0, starpu_cuda_get_local_stream()>>>(ePot, ePot_worker);
    cudaStreamSynchronize(starpu_cuda_get_local_stream());
}

extern "C" void ePot_init_cuda_func(void *descr[], void *cl_arg){
    real_t *ePot = (real_t *)STARPU_VARIABLE_GET_PTR(descr[0]);
	cudaMemsetAsync(ePot, 0, sizeof(real_t), starpu_cuda_get_local_stream());
}
