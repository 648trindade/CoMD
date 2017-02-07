/// \file
/// Computes forces for the 12-6 Lennard Jones (LJ) potential.
///
/// The Lennard-Jones model is not a good representation for the
/// bonding in copper, its use has been limited to constant volume
/// simulations where the embedding energy contribution to the cohesive
/// energy is not included in the two-body potential
///
/// The parameters here are taken from Wolf and Phillpot and fit to the
/// room temperature lattice constant and the bulk melt temperature
/// Ref: D. Wolf and S.Yip eds. Materials Interfaces (Chapman & Hall
///      1992) Page 230.
///
/// Notes on LJ:
///
/// http://en.wikipedia.org/wiki/Lennard_Jones_potential
///
/// The total inter-atomic potential energy in the LJ model is:
///
/// \f[
///   E_{tot} = \sum_{ij} U_{LJ}(r_{ij})
/// \f]
/// \f[
///   U_{LJ}(r_{ij}) = 4 \epsilon
///           \left\{ \left(\frac{\sigma}{r_{ij}}\right)^{12}
///           - \left(\frac{\sigma}{r_{ij}}\right)^6 \right\}
/// \f]
///
/// where \f$\epsilon\f$ and \f$\sigma\f$ are the material parameters in the potential.
///    - \f$\epsilon\f$ = well depth
///    - \f$\sigma\f$   = hard sphere diameter
///
///  To limit the interation range, the LJ potential is typically
///  truncated to zero at some cutoff distance. A common choice for the
///  cutoff distance is 2.5 * \f$\sigma\f$.
///  This implementation can optionally shift the potential slightly
///  upward so the value of the potential is zero at the cuotff
///  distance.  This shift has no effect on the particle dynamics.
///
///
/// The force on atom i is given by
///
/// \f[
///   F_i = -\nabla_i \sum_{jk} U_{LJ}(r_{jk})
/// \f]
///
/// where the subsrcipt i on the gradient operator indicates that the
/// derivatives are taken with respect to the coordinates of atom i.
/// Liberal use of the chain rule leads to the expression
///
/// \f{eqnarray*}{
///   F_i &=& - \sum_j U'_{LJ}(r_{ij})\hat{r}_{ij}\\
///       &=& \sum_j 24 \frac{\epsilon}{r_{ij}} \left\{ 2 \left(\frac{\sigma}{r_{ij}}\right)^{12}
///               - \left(\frac{\sigma}{r_{ij}}\right)^6 \right\} \hat{r}_{ij}
/// \f}
///
/// where \f$\hat{r}_{ij}\f$ is a unit vector in the direction from atom
/// i to atom j.
/// 
///

#include "ljForce.h"

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <omp.h>

#include "constants.h"
#include "mytype.h"
#include "parallel.h"
#include "linkCells.h"
#include "memUtils.h"
#include "CoMDTypes.h"
#include "starpu_code.h"

#define POT_SHIFT 1.0

/// Derived struct for a Lennard Jones potential.
/// Polymorphic with BasePotential.
/// \see BasePotential
typedef struct LjPotentialSt
{
   real_t cutoff;          //!< potential cutoff distance in Angstroms
   real_t mass;            //!< mass of atoms in intenal units
   real_t lat;             //!< lattice spacing (angs) of unit cell
   char latticeType[8];    //!< lattice type, e.g. FCC, BCC, etc.
   char  name[3];	   //!< element name
   int	 atomicNo;	   //!< atomic number  
   int  (*force)(SimFlat* s); //!< function pointer to force routine
   void (*print)(FILE* file, BasePotential* pot);
   void (*destroy)(BasePotential** pot); //!< destruction of the potential
   real_t sigma;
   real_t epsilon;
} LjPotential;

static int ljForce(SimFlat* s);
static void ljPrint(FILE* file, BasePotential* pot);

void ljDestroy(BasePotential** inppot)
{
   if ( ! inppot ) return;
   LjPotential* pot = (LjPotential*)(*inppot);
   if ( ! pot ) return;
   comdFree(pot);
   *inppot = NULL;

   return;
}

/// Initialize an Lennard Jones potential for Copper.
BasePotential* initLjPot(void)
{
   LjPotential *pot = (LjPotential*)comdMalloc(sizeof(LjPotential));
   pot->force = ljForce;
   pot->print = ljPrint;
   pot->destroy = ljDestroy;
   pot->sigma = 2.315;	                  // Angstrom
   pot->epsilon = 0.167;                  // eV
   pot->mass = 63.55 * amuToInternalMass; // Atomic Mass Units (amu)

   pot->lat = 3.615;                      // Equilibrium lattice const in Angs
   strcpy(pot->latticeType, "FCC");       // lattice type, i.e. FCC, BCC, etc.
   pot->cutoff = 2.5*pot->sigma;          // Potential cutoff in Angs

   strcpy(pot->name, "Cu");
   pot->atomicNo = 29;

   return (BasePotential*) pot;
}

void ljPrint(FILE* file, BasePotential* pot)
{
   LjPotential* ljPot = (LjPotential*) pot;
   fprintf(file, "  Potential type   : Lennard-Jones\n");
   fprintf(file, "  Species name     : %s\n", ljPot->name);
   fprintf(file, "  Atomic number    : %d\n", ljPot->atomicNo);
   fprintf(file, "  Mass             : "FMT1" amu\n", ljPot->mass / amuToInternalMass); // print in amu
   fprintf(file, "  Lattice Type     : %s\n", ljPot->latticeType);
   fprintf(file, "  Lattice spacing  : "FMT1" Angstroms\n", ljPot->lat);
   fprintf(file, "  Cutoff           : "FMT1" Angstroms\n", ljPot->cutoff);
   fprintf(file, "  Epsilon          : "FMT1" eV\n", ljPot->epsilon);
   fprintf(file, "  Sigma            : "FMT1" Angstroms\n", ljPot->sigma);
}

int ljForce(SimFlat* s)
{
   LjPotential* pot = (LjPotential *) s->pot;
   real_t sigma = pot->sigma;
   real_t epsilon = pot->epsilon;
   real_t rCut = pot->cutoff;
   real_t rCut2 = rCut*rCut;

   // zero forces and energy
   real_t ePot = 0.0;
   s->ePotential = 0.0;
   int fSize = s->boxes->nTotalBoxes*MAXATOMS;
   #pragma omp parallel for
   for (int ii=0; ii<fSize; ++ii)
   {
      zeroReal3(s->atoms->f[ii]);
      s->atoms->U[ii] = 0.;
   }
   
   real_t s6 = sigma*sigma*sigma*sigma*sigma*sigma;

   real_t rCut6 = s6 / (rCut2*rCut2*rCut2);
   real_t eShift = POT_SHIFT * rCut6 * (rCut6 - 1.0);

   int nNbrBoxes = 27;

   // Uma task por box / laço
   NTASKS = s->boxes->nLocalBoxes;

   /* Todo dado registrado por um handle poderá(ia) ser usado sem forçar
      serialização das tasks a partir desse ponto */
   starpu_data_set_default_sequential_consistency_flag(0);

   // Registra os handles necessários
   starpu_vector_data_register(&nAtoms_handle, STARPU_MAIN_RAM,
      (uintptr_t)s->boxes->nAtoms, s->boxes->nTotalBoxes, sizeof(int));
   starpu_vector_data_register(&r_handle, STARPU_MAIN_RAM, 
      (uintptr_t)s->atoms->r, fSize * 3, 
      sizeof(real_t));
   starpu_vector_data_register(&f_handle, STARPU_MAIN_RAM, 
      (uintptr_t)s->atoms->f, s->boxes->nLocalBoxes * MAXATOMS * 3,
      sizeof(real_t));
   starpu_vector_data_register(&U_handle, STARPU_MAIN_RAM, 
      (uintptr_t)s->atoms->U, s->boxes->nLocalBoxes * MAXATOMS, sizeof(real_t));
   starpu_variable_data_register(&ePot_handle, STARPU_MAIN_RAM, 
      (uintptr_t)&ePot, sizeof(real_t));
   starpu_vector_data_register(&nbrBoxes_handle, STARPU_MAIN_RAM, 
      (uintptr_t)s->boxes->nbrBoxes[0], s->boxes->nLocalBoxes * 27, 
      sizeof(s->boxes->nbrBoxes[0][0]));
   
   // Dados serão particionados pelo número de tasks
   data_filter.nchildren = NTASKS;
   starpu_data_partition(nbrBoxes_handle, &data_filter);
   starpu_data_partition(U_handle, &data_filter);
   starpu_data_partition(f_handle, &data_filter);

   // Dita as funções que definem inicialização de instâncias e redução de ePot
   starpu_data_set_reduction_methods(ePot_handle, &ePot_redux_codelet, &ePot_init_codelet);

   // Copia paramêtros
   struct params params = {
      .s6 = s6, 
      .eShift = eShift,
      .epsilon = epsilon, 
      .rCut2 = rCut2,
      .nNbrBoxes = nNbrBoxes,
      .nLocalBoxes = s->boxes->nLocalBoxes
   };

   // loop over local boxes
   //#pragma omp parallel for reduction(+:ePot)
   for (int id = 0; id < NTASKS; id++)
   {
      // Coração do laço retirado e movido pra cpu_func()
      // Definição da task
      struct starpu_task *task = starpu_task_create();
      task->cl = &cl;
      task->cl_arg = &params;
      task->cl_arg_size = sizeof(struct params);
      task->synchronous = 0;
      // Atribui handles para a task
      task->handles[0] = starpu_data_get_sub_data(nbrBoxes_handle, 1, id);
      task->handles[1] = nAtoms_handle;
      task->handles[2] = r_handle;
      task->handles[3] = starpu_data_get_sub_data(f_handle, 1, id);
      task->handles[4] = starpu_data_get_sub_data(U_handle, 1, id);
      task->handles[5] = ePot_handle;

      // Submete a task assincronamente
      int ret = starpu_task_submit(task);
      STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
   } // loop over local boxes in system

   starpu_task_wait_for_all();
   starpu_data_unpartition(nbrBoxes_handle, STARPU_MAIN_RAM);
   starpu_data_unpartition(U_handle, STARPU_MAIN_RAM);
   starpu_data_unpartition(f_handle, STARPU_MAIN_RAM);

   // Desregistra todos os handles
   starpu_data_unregister(nAtoms_handle);
   starpu_data_unregister(r_handle);
   starpu_data_unregister(f_handle);
   starpu_data_unregister(U_handle);
   starpu_data_unregister(ePot_handle);
   starpu_data_unregister(nbrBoxes_handle);

   ePot = ePot*4.0*epsilon;
   s->ePotential = ePot;

   //starpu_codelet_display_stats(&cl);

   return 0;
}

// Laço principal do ljForce (itera sobre todas as caixas) 
void cpu_func(void *buffers[], void *cl_arg){
    
    // Angariando parâmetros
    struct params *params = cl_arg;
    real_t       s6 = params->s6;
    real_t   eShift = params->eShift;
    real_t  epsilon = params->epsilon;
    real_t    rCut2 = params->rCut2;
    int   nNbrBoxes = params->nNbrBoxes;
    int nLocalBoxes = params->nLocalBoxes;
    
    // Angariando buffers
    int* nbrBoxes = (   int*) STARPU_VECTOR_GET_PTR(buffers[0]);
    int*   nAtoms = (   int*) STARPU_VECTOR_GET_PTR(buffers[1]);
    real3*      r = ( real3*) STARPU_VECTOR_GET_PTR(buffers[2]);
    real3*      f = ( real3*) STARPU_VECTOR_GET_PTR(buffers[3]);
    real_t*     U = (real_t*) STARPU_VECTOR_GET_PTR(buffers[4]);
    real_t*  ePot = (real_t*) STARPU_VARIABLE_GET_PTR(buffers[5]);
    
    // Angariando offsets e números de elementos
    size_t nbrBoxes_offset = (size_t) STARPU_VECTOR_GET_OFFSET(buffers[0]);
    size_t nbrBoxes_nx = (size_t) STARPU_VECTOR_GET_NX(buffers[0]);
    size_t iOff_offset = (size_t) STARPU_VECTOR_GET_OFFSET(buffers[4]);
    size_t f_nx = (size_t) STARPU_VECTOR_GET_NX(buffers[3]);
    size_t U_nx = (size_t) STARPU_VECTOR_GET_NX(buffers[4]);

    // Conferindo se offsets e tamanhos estão dentro do esperado
    STARPU_ASSERT((nbrBoxes_offset / sizeof(int)) % nNbrBoxes == 0);
    STARPU_ASSERT(nbrBoxes_nx % nNbrBoxes == 0);
    STARPU_ASSERT((iOff_offset / sizeof(real_t)) % MAXATOMS == 0);
    STARPU_ASSERT(U_nx % MAXATOMS == 0);
    STARPU_ASSERT((f_nx / 3) % MAXATOMS == 0);
    
    // Calculando offsets e tamanhos reais
    nbrBoxes_offset /= nNbrBoxes * sizeof(int);
    nbrBoxes_nx     /= nNbrBoxes;
    iOff_offset     /= sizeof(real_t);

    for (int iBox = nbrBoxes_offset; iBox < nbrBoxes_offset + nbrBoxes_nx && iBox < nLocalBoxes; iBox++){

      int nIBox = nAtoms[iBox];
      
      // loop over neighbors of iBox
      for (int jTmp=0; jTmp<nNbrBoxes; jTmp++){
            // Acessa nbrBoxes de modo contíguo
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
                     *ePot = *ePot + 0.5*eLocal;

                     // different formulation to avoid sqrt computation
                     real_t fr = - 4.0*epsilon*r6*r2*(12.0*r6 - 6.0);
                     for (int m=0; m<3; m++)
                           f[task_iOff][m] -= dr[m]*fr;
                  }
               } // loop over atoms in jBox
         } // loop over atoms in iBox
      } // loop over neighbor boxes
    }
}

void ePot_redux_cpu_func(void *descr[], void *cl_arg)
{
	real_t *ePot = (real_t *)STARPU_VARIABLE_GET_PTR(descr[0]);
	real_t *ePot_partial = (real_t *)STARPU_VARIABLE_GET_PTR(descr[1]);

      *ePot = *ePot + *ePot_partial;
}

void ePot_init_cpu_func(void *descr[], void *cl_arg)
{
	real_t *ePot = (real_t *)STARPU_VARIABLE_GET_PTR(descr[0]);
	*ePot = 0.0;
}