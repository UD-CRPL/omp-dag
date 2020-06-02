/*---------------------------------------------------------------------------*/
/*!
 * \file   dimensions.c
 * \author Robert Searles
 * \date   Tue Sep 11 16:38:00 EST 2018
 * \brief  Dimensions implementation file
 * \note   
 */
/*---------------------------------------------------------------------------*/

#include "dimensions.h"

/*===========================================================================*/
/*--- Malloc function ---*/

real* malloc_host_real( size_t n )
{
  Assert( n+1 >= 1 );
  real* result = (real*)malloc( n * sizeof(real) );
  Assert( result );
  return result;
}
