/*---------------------------------------------------------------------------*/
/*!
 * \file   simulation.h
 * \author Robert Searles
 * \date   Tue Sep 11 17:21:00 EST 2018
 * \brief  Simulation parent class for in-gridcell algorithms
 * \note   
 */
/*---------------------------------------------------------------------------*/

#ifndef _simulation_h_
#define _simulation_h_

#include "dimensions.h"
#include "arguments.h"
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

/*===========================================================================*/
/*--- Radiation Transport class ---*/

class Simulation
{
 public:
  /*=============================================*/
  /*=== MANDATORY FUNCTIONS ===*/
  /*=============================================*/

  /*--- Initialization ---*/
  virtual void init(){}

  /*--- Run ---*/
  /*--- This should run your wavefront sweep component & in-gridcell computation ---*/
  virtual void run(){}
  
  /*--- Print Output ---*/
  virtual void print(){}

 protected:
  /*===========================================================================*/
  /*--- Malloc function ---*/

  real* malloc_host_real( size_t n )
  {
    Assert( n+1 >= 1 );
    real* result = (real*)malloc( n * sizeof(real) );
    Assert( result );
    return result;
  }

  int* malloc_host_int( size_t n )
  {
    Assert( n+1 >= 1 );
    int* result = (int*)malloc( n * sizeof(int) );
    Assert( result );
    return result;
  }

  /*=============================================*/
  /*=== ALGORITHM-SPECIFIC FUNCTIONS ===*/
  /*=============================================*/

  // FILL THESE IN WHEN ADDING YOUR ALGORITHMS

  /*=============================================*/
  /*=== DATA ===*/
  /*=============================================*/

  /*--- Output array ---*/
  real* __restrict__ output;

  /*--- Simulation dimensions ---*/
  Dimensions dims;
};

#endif /*--- _simulation_h_ ---*/

/*---------------------------------------------------------------------------*/
