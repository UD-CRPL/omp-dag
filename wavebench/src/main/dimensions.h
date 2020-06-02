/*---------------------------------------------------------------------------*/
/*!
 * \file   dimensions.h
 * \author Robert Searles
 * \date   Tue Sep 11 16:38:00 EST 2018
 * \brief  Dimensions header file
 * \note   
 */
/*---------------------------------------------------------------------------*/

#ifndef _dims_h_
#define _dims_h_

#pragma once
#include <stdlib.h>
#include <stddef.h>

typedef double real;

/*===========================================================================*/
/*--- Dimensions struct ---*/

typedef struct
{
  /*--- Grid spatial dimensions ---*/
  int ncell_x;
  int ncell_y;
  int ncell_z;

  /*--- Number of in-gridcell components ---*/
  int ncomponents;

  /*--- Number of timesteps to run ---*/
  int niterations;

  /*--- Algorithm Identifier ---*/
  int alg_id;
} Dimensions;

#endif
