/*---------------------------------------------------------------------------*/
/*!
 * \file   wavebench.cpp
 * \author Robert Searles
 * \date   Wed Mar 8 16:32:00 EST 2018
 * \brief  Wavefront benchmark sweep application
 * \note   
 */
/*---------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h> /* memset */
#include <sys/time.h>

#include "arguments.h"
#include "dimensions.h"
#include "simulation.h"
#include "wavebench.h"
#include "radiation_transport.h"
#include "local_sequence_alignment.h"
#include "gauss_seidel.h"

/*===========================================================================*/
/*--- Parse command line arguments ---*/

Dimensions parse_args( int argc, char** argv,bool& print )
{
  printf("=======================================================\n"
	 "Wavefront Benchmark\n"
	 "Robert Searles\n"
	 "University of Delaware\n"
	 "Department of Computer and Information Sciences\n\n"
	 "Contact: rsearles@udel.edu\n"
	 "=======================================================\n");

  Arguments args = Arguments_null();
  Arguments_create( &args, argc, argv );

  /*--- Help menu ---*/
  if (Arguments_exists(&args, "--help") || Arguments_exists(&args, "-help")) {
    printf("usage: wavebench [options]\n"); // FILL THIS IN LATER
  }

  printf("\nParsing command line arguments...\n\n");
  
  /*--- Consume problem size args ---*/
  Dimensions dims;
  dims.ncell_x = (int) Arguments_consume_int_or_default( &args, "--ncell_x",  5 );
  dims.ncell_y = (int) Arguments_consume_int_or_default( &args, "--ncell_y",  5 );
  dims.ncell_z = (int) Arguments_consume_int_or_default( &args, "--ncell_z",  5 );
  dims.ncomponents = (int) Arguments_consume_int_or_default( &args, "--ncomp",  32 );
  dims.niterations = (int) Arguments_consume_int_or_default( &args, "--niterations", 1 );
  print=((int)Arguments_consume_int_or_default( &args, "--print", 1 ))!=0;

  Insist( dims.ncell_x > 0 ? "Invalid ncell_x supplied." : 0 );
  Insist( dims.ncell_y > 0 ? "Invalid ncell_y supplied." : 0 );
  Insist( dims.ncell_z > 0 ? "Invalid ncell_z supplied." : 0 );
  Insist( dims.ncomponents > 0 ? "Invalid ncomp supplied." : 0 );
  Insist( dims.niterations > 0 ? "Invalid niterations supplied." : 0 );

  /*--- Consume Algorithm Identifier ---*/
  dims.alg_id = (int) Arguments_consume_int_or_default( &args, "--alg",  0 );
  Insist( dims.alg_id <= 2 ? "Invalid alg_id supplied." : 0 );

  Insist( Arguments_are_all_consumed( &args )
	  ? "Invalid argument detected." : 0 );

  printf("Algorithm ID: %d\n\n", dims.alg_id);

  return dims;
}

/*===========================================================================*/
/*--- Main simulation ---*/

int main( int argc, char** argv)
{
  /*--- Parse command line arguments ---*/
	bool print=false;
  Dimensions dims = parse_args(argc, argv,print);

  /*--- Timer ---*/
  float runtime;

  Simulation* simulation;
  /*--- Create simulation structure ---*/
  if (dims.alg_id == 0) {
    /*--- Print simulation dimensions ---*/
    printf("-------------------------------------------------------\n"
	   "Simulation dimensions:\n"
	   "-------------------------------------------------------\n"
	   "niterations: %d\n"
	   "ncell_x: %d\n"
	   "ncell_y: %d\n"
	   "ncell_z: %d\n"
	   "ncomponents: %d\n\n",
	   dims.niterations,
	   dims.ncell_x,
	   dims.ncell_y,
	   dims.ncell_z,
	   dims.ncomponents
	   );

    simulation = new RT(dims);
  }
  else if (dims.alg_id == 1) {
    simulation = new LSA(dims);
    printf("-------------------------------------------------------\n"
	   "Simulation dimensions:\n"
	   "-------------------------------------------------------\n"
	   "ncell_x: %d\n"
	   "ncell_y: %d\n",
	   static_cast<LSA*>(simulation)->r(),
	   static_cast<LSA*>(simulation)->c()
	   );
  }
  else if (dims.alg_id == 2) {
    /*--- Print simulation dimensions ---*/
    printf("-------------------------------------------------------\n"
	   "Simulation dimensions:\n"
	   "-------------------------------------------------------\n"
	   "niterations: %d\n"
	   "ncell_x: %d\n"
	   "ncell_y: %d\n"
	   "ncell_z: %d\n"
	   "ncomponents: %d\n\n",
	   dims.niterations,
	   dims.ncell_x,
	   dims.ncell_y,
	   dims.ncell_z,
	   dims.ncomponents
	   );

    simulation = new GS(dims);
  }
  else {
    printf("ERROR: Invalid algorithm identifier\n");
    exit(-1);
  }

  /*--- Start timer ---*/
  simulation->init();
  timeval ts, te;
  gettimeofday(&ts, NULL);

  /*--- Initialize for appropriate algorithm ---*/

  /*--- Perform sweep ---*/
  int iteration;
  for (iteration=0; iteration<dims.niterations; ++iteration) {
    simulation->run();
  }

  /*--- End timer ---*/
  gettimeofday(&te, NULL);

  /*--- Print output array ---*/
  printf("-------------------------------------------------------\n"
	 "Results:\n"
	 "-------------------------------------------------------\n");
  if(print)
	  simulation->print();

  runtime = (float)((te.tv_sec * 1000000 + te.tv_usec) 
		    - (ts.tv_sec * 1000000 + ts.tv_usec))/1000000;

  /*--- Print timing information ---*/
  fprintf (stderr, "Sweep time: %f seconds.\n", runtime);
  
  /*--- Destroy simulation structure ---*/
  // Destructor handles this
  //simulation->destroy();

  return 0;
}

/*---------------------------------------------------------------------------*/
