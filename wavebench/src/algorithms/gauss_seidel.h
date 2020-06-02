/*---------------------------------------------------------------------------*/
/*!
 * \file   gauss_seidel.h
 * \author Robert Searles
 * \date   Wed Jan 2 16:52:00 EST 2019
 * \brief  Gauss Seidel in-gridcell algorithm
 * \note   
 */
/*---------------------------------------------------------------------------*/

#ifndef _gauss_seidel_h_
#define _gauss_seidel_h_

#include "dimensions.h"
#include "simulation.h"

/*===========================================================================*/
/*--- Local Sequence Alignment class ---*/

class GS : public Simulation
{
 public:
  /*=============================================*/
  /*=== MANDATORY FUNCTIONS ===*/
  /*=============================================*/

  /*--- Constructor ---*/
  GS() {
    printf("Default Constructor. DO NOT USE. This does not pass along simulation dimensions!\n\n");
    exit(0);
  }  

  GS(Dimensions d);

  /*--- Initialization ---*/
  void init();

  /*--- Run ---*/
  /*--- This should run your wavefront sweep component & in-gridcell computation ---*/
  void run();

  /*--- Print Output ---*/
  void print();

 private:
  /*=============================================*/
  /*=== ALGORITHM-SPECIFIC FUNCTIONS ===*/
  /*=============================================*/
  void solve(int nu, int ncellx, int ncelly, 
	           real* __restrict__ vo, 
	     const real* __restrict__ vi, 
	     const real* __restrict__ an, 
	     const real* __restrict__ as, 
	     const real* __restrict__ ae, 
	     const real* __restrict__ aw);

  /*===========================================================================*/
  /*---5-point stencil operation applied at a gridcell---*/

#pragma acc routine vector
  void process_cell(int ix, int iy, int nu, int ncellx, int ncelly,
		          real* __restrict__ vo,
		    const real* __restrict__ vi,
		    const real* __restrict__ an,
		    const real* __restrict__ as,
		    const real* __restrict__ ae,
		    const real* __restrict__ aw);

  /*=============================================*/
  /*=== DATA ===*/
  /*=============================================*/

  /*--- Simulation Dimensions ---*/
  int v_size, a_size, ncellx, ncelly, nu;

  /*--- Simulation Data ---*/
  real* __restrict__ v1;
  real* __restrict__ v2;
  real* __restrict__ an;
  real* __restrict__ as;
  real* __restrict__ ae;
  real* __restrict__ aw;

  /*--- Output Array ---*/
  real* __restrict__ vfinal;
};

#endif /*--- _gauss_seidel_h_ ---*/

/*---------------------------------------------------------------------------*/
