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

   // Handle for s->boxes->nAtoms
   data_handle *nAtoms_handle = create_and_register_vector_handle(
      s->boxes->nAtoms,
      s->boxes->nTotalBoxes,
      sizeof(int)
   );

   // Handle for s->atoms->r
   data_handle *r_handle = create_and_register_vector_handle(
      s->atoms->r,
      fSize,
      sizeof(real3)
   );

    // Handle for s->atoms->f
   data_handle *f_handle = create_and_register_vector_handle(
      s->atoms->f,
      fSize,
      sizeof(real3)
   );

    // Handle for s->atoms->U
   data_handle *U_handle = create_and_register_vector_handle(
      s->atoms->U,
      fSize,
      sizeof(real_t)
   );

    // Handle for ePot
   data_handle *ePot_handle = create_and_register_variable_handle(
        &ePot,
        sizeof(ePot)
   );

   // Handle array for future unregistration and deallocation (line 252)
   data_handle* created_handles[5 + s->boxes->nLocalBoxes];
   created_handles[0] = nAtoms_handle;
   created_handles[1] = r_handle;
   created_handles[2] = f_handle;
   created_handles[3] = U_handle;
   created_handles[4] = ePot_handle;

   // loop over local boxes
   //#pragma omp parallel for reduction(+:ePot)
   for (int iBox=0; iBox<s->boxes->nLocalBoxes; iBox++)
   {
      // coração do laço retirado e movido pra starpu_code.c:cpu_func()

      // copia paramêtros
      struct params params = {
         .s6 = s6, 
         .eShift = eShift,
         .epsilon = epsilon, 
         .rCut2 = rCut2, 
         .iBox = iBox,
         .nNbrBoxes = nNbrBoxes
      };

      // Handle for s->boxes->nbrBoxes[iBox]
      data_handle *jBoxes_handle = create_and_register_vector_handle(
         s->boxes->nbrBoxes[iBox],
         nNbrBoxes,
         sizeof(s->boxes->nbrBoxes[iBox][0])
      );

      // copy new handle to handle array
      created_handles[5 + iBox] = jBoxes_handle;

      // create a short handle array for task use
      data_handle* handles[] = {
         jBoxes_handle,
         nAtoms_handle,
         r_handle,
         f_handle,
         U_handle,
         ePot_handle
      };

      // create the task for current iteration
      create_and_start_task(handles, &params, sizeof(params));
   } // loop over local boxes in system

   // Unregister all data handles and free memory (and wait for tasks)
   unregister_and_destroy_data_handles(created_handles, 5 + s->boxes->nLocalBoxes);

   ePot = ePot*4.0*epsilon;
   s->ePotential = ePot;

   return 0;
}