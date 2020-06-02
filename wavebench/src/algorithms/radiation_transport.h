/*---------------------------------------------------------------------------*/
/*!
 * \file   radiation_transport.h
 * \author Robert Searles
 * \date   Tue Sep 11 16:11:00 EST 2018
 * \brief  Radiation Transport in-gridcell algorithm
 * \note   
 */
/*---------------------------------------------------------------------------*/

#ifndef _radiation_transport_h_
#define _radiation_transport_h_

#include "dimensions.h"
#include "simulation.h"

/*===========================================================================*/
/*--- Radiation Transport class ---*/

class RT : public Simulation
{
 public:
  /*=============================================*/
  /*=== MANDATORY FUNCTIONS ===*/
  /*=============================================*/

  /*--- Constructor ---*/
  RT() {
    printf("Default Constructor. DO NOT USE. This does not pass along simulation dimensions!\n\n");
    exit(0);
  }  

  RT(Dimensions d);

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

  /*--- Solver ---*/
  /*--- (This is called for each gridcell). This will contain your algorithm. ---*/
#pragma acc routine vector
  void solve(int ix,
	     int iy,
	     int iz,
	     int octant);
  
  /*--- compute routine ---*/
#pragma acc routine seq
  void compute(int ix, int iy, int iz, int ie, int ia, int octant);

  /*--- Quantities_init_face routine ---*/
#pragma acc routine seq
  real Quantities_init_face(int ia, int ie, int iu, int scalefactor_space, int octant);

  /*--- Quantities_scalefactor_space routine ---*/
#pragma acc routine seq
  int Quantities_scalefactor_space(int ix, int iy, int iz);

#pragma acc routine seq
  int Quantities_scalefactor_energy(int ic);

  /*=============================================*/
  /*=== DATA ===*/
  /*=============================================*/

  /*--- Simulation dimensions ---*/
  int NE;
  int NU = 4;
  int NM = 4;
  int NA = 32;
  int NOCTANT = 8;

  /*--- Face arrays used for upstream data dependencies ---*/
  real* __restrict__ facexy;
  real* __restrict__ facexz;
  real* __restrict__ faceyz;
  real* __restrict__ local;
  real* __restrict__ input;
  real* __restrict__ m_from_a;
};

#endif /*--- _radiation_transport_h_ ---*/

/*---------------------------------------------------------------------------*/
