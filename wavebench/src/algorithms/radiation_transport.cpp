/*---------------------------------------------------------------------------*/
/*!
 * \file   radiation_transport.cpp
 * \author Robert Searles
 * \date   Tue Sep 11 16:11:00 EST 2018
 * \brief  Radiation Transport in-gridcell algorithm
 * \note   
 */
/*---------------------------------------------------------------------------*/

#include "dimensions.h"
#include "radiation_transport.h"
#include <stdio.h>

/*=============================================*/
/*=== MANDATORY FUNCTIONS ===*/
/*=============================================*/

/*--- Constructor ---*/
RT::RT(Dimensions d) {
  printf("Creating Radiation Transport Simulation\n\n");

  dims = d;
  NE = dims.ncomponents;

#pragma acc enter data copyin(this)

  /*--- Array Sizes ---*/
  int facexy_size = dims.ncell_x * dims.ncell_y * NE * NA * NU * NOCTANT;
  int facexz_size = dims.ncell_x * dims.ncell_z * NE * NA * NU * NOCTANT;
  int faceyz_size = dims.ncell_y * dims.ncell_z * NE * NA * NU * NOCTANT;
  int local_size = dims.ncell_x * dims.ncell_y * NE * NA * NU * NOCTANT;
  int v_size = NM * NA * NOCTANT;
  int input_size = dims.ncell_x * dims.ncell_y * dims.ncell_z * NE * NU * NM;
  int output_size = dims.ncell_x * dims.ncell_y * dims.ncell_z * NE * NU * NM;

  /*--- Allocate Arrays ---*/
  facexy = (real*)malloc_host_real(facexy_size);
  facexz = (real*)malloc_host_real(facexz_size);
  faceyz = (real*)malloc_host_real(faceyz_size);
  local = (real*)malloc_host_real(local_size);
  m_from_a = (real*)malloc_host_real(v_size);
  input = (real*)malloc_host_real(input_size);
  output = (real*)malloc_host_real(output_size);

  /*--- Input Array (this is normally where we'd read in data) ---*/
  for (int iz=0; iz<dims.ncell_z; ++iz)
    for (int iy=0; iy<dims.ncell_y; ++iy)
      for (int ix=0; ix<dims.ncell_x; ++ix)
      for( int ie=0; ie<NE; ++ie )
      for( int iu=0; iu<NU; ++iu )
      for ( int im=0; im<NM; ++im )
	{
	  input[ im + NM * ( 
		 iu + NU * (
		 ie + NE * (
		 ix + dims.ncell_x * (
		 iy + dims.ncell_y * (
		 iz + dims.ncell_z * (
				      0))))))] = (real) (Quantities_scalefactor_space(ix, iy, iz) * (real) Quantities_scalefactor_energy(ie) * iu) + (real)(im*iu);
	}

  /*--- moments/angles conversion arrays ---*/
  for (int octant=0; octant<NOCTANT; ++octant)
    for (int im=0; im<NM; ++im)
      for (int ia=0; ia<NA; ++ia)
	{
	  m_from_a[im + NM * (
		   ia + NA * (
		   octant + NOCTANT * (
				       0)))] = (real)(im+1) + (1.0/(octant+1));
	}

  /*--- Copy/Create data to/on the GPU ---*/
#pragma acc enter data copyin(dims, input[:input_size],\
			      m_from_a[:v_size]), \
                                     create(facexy[:facexy_size], \
					    facexz[:facexz_size], \
					    faceyz[:faceyz_size], \
					    local[:local_size],	\
					    output[:output_size])

}

/*--- Initialization ---*/
void RT::init(){

  /*--- Declarations ---*/
  int octant = 0;
  int ix = 0;
  int iy = 0;
  int iz = 0;
  int ie = 0;
  int iu = 0;
  int ia = 0;
  int im = 0;

  /*--- Dimensions ---*/
  int dim_x = dims.ncell_x;
  int dim_y = dims.ncell_y;
  int dim_z = dims.ncell_z;

  /*--- Array Sizes ---*/
  int facexy_size = dims.ncell_x * dims.ncell_y * NE * NA * NU * NOCTANT;
  int facexz_size = dims.ncell_x * dims.ncell_z * NE * NA * NU * NOCTANT;
  int faceyz_size = dims.ncell_y * dims.ncell_z * NE * NA * NU * NOCTANT;
  int local_size = dims.ncell_x * dims.ncell_y * NE * NA * NU * NOCTANT;
  int v_size = NM * NA * NOCTANT;
  int input_size = dims.ncell_x * dims.ncell_y * dims.ncell_z * NE * NU * NM;
  int output_size = dims.ncell_x * dims.ncell_y * dims.ncell_z * NE * NU * NM;

  /*--- Initialize Faces and Result Arrays ---*/
#pragma acc parallel present(facexy[:facexy_size], \
			     facexz[:facexz_size], \
			     faceyz[:faceyz_size], \
                             m_from_a[:v_size], \
			     local[:local_size], \
                             input[:input_size], \
			     output[:output_size], dims, this)
  {

  /*--- Initialize output array ---*/
#pragma acc loop independent gang collapse(3)
  for (iz=0; iz<dim_z; ++iz)
    for (iy=0; iy<dim_y; ++iy)
      for (ix=0; ix<dim_x; ++ix)
#pragma acc loop independent vector collapse(2)
      for( ie=0; ie<NE; ++ie )
      for( iu=0; iu<NU; ++iu )
	{
	  output[im + NM * ( 
		 iu + NU * (
		 ie + NE * (
		 ix + dim_x * (
		 iy + dim_y * (
		 iz + dim_z * (
			       0))))))] = (real)0.0;
	}

  /*--- Initialize faces ---*/
  
  /*--- The semantics of the face arrays are as follows.
         On entering a cell for a solve at the gridcell level,
         the face array is assumed to have a value corresponding to
         "one cell lower" in the relevant direction.
         On leaving the gridcell solve, the face has been updated
         to have the value at that gridcell.
     ---*/

  /*--- XY Face ---*/
#pragma acc loop independent gang collapse(3)
  for( octant=0; octant<NOCTANT; ++octant )
    for (iy=0; iy<dim_y; ++iy)
      for (ix=0; ix<dim_x; ++ix)
#pragma acc loop independent vector collapse(3)
      for( ie=0; ie<NE; ++ie )
      for( iu=0; iu<NU; ++iu )
      for( ia=0; ia<NA; ++ia )
	  {
	    iz = -1;
	    int scalefactor_space = Quantities_scalefactor_space(ix, iy, iz);

	    facexy[ia + NA * (
		   iu + NU * (
		   ie + NE * (
		   ix + dim_x * (
		   iy + dim_y * (
		   octant + NOCTANT * (
				       0 ))))))]
	      = Quantities_init_face(ia, ie, iu, scalefactor_space, octant);
	  }

  /*--- XZ Face ---*/
#pragma acc loop independent gang collapse(3)
  for( octant=0; octant<NOCTANT; ++octant )
    for (iz=0; iz<dim_z; ++iz)
      for (ix=0; ix<dim_x; ++ix)
#pragma acc loop independent vector collapse(3)
	for( ie=0; ie<NE; ++ie )
	for( iu=0; iu<NU; ++iu )
	for( ia=0; ia<NA; ++ia )
	  {
	    iy = -1;
	    int scalefactor_space = Quantities_scalefactor_space(ix, iy, iz);

	    facexz[ia + NA * (
		   iu + NU * (
		   ie + NE * (
		   ix + dim_x * (
		   iz + dim_z * (
		   octant + NOCTANT * (
				       0 ))))))]
	      = Quantities_init_face(ia, ie, iu, scalefactor_space, octant);
	  }

  /*--- YZ Face ---*/
#pragma acc loop independent gang collapse(3)
  for( octant=0; octant<NOCTANT; ++octant )
    for (iz=0; iz<dim_z; iz++)
      for (iy=0; iy<dim_y; iy++)
#pragma acc loop independent vector collapse(3)
	for( ie=0; ie<NE; ++ie )
	for( iu=0; iu<NU; ++iu )
	for( ia=0; ia<NA; ++ia )
	  {
	    ix = -1;
	    int scalefactor_space = Quantities_scalefactor_space(ix, iy, iz);

	    faceyz[ia + NA * (
		   iu + NU * (
		   ie + NE * (
		   iy + dim_y * (
		   iz + dim_z * (
		   octant + NOCTANT * (
				       0 ))))))]
	      = Quantities_init_face(ia, ie, iu, scalefactor_space, octant);
	  }
  } /*--- #pragma acc parallel ---*/
}

/*--- Run ---*/
/*--- This should run your wavefront sweep component & in-gridcell computation ---*/
void RT::run(){
  printf("sweeping...\n\n");
  /*--- Declarations ---*/
  int octant = 0;
  int wavefront = 0;
  int ix = 0;
  int iy = 0;
  int iz = 0;

  /*--- Dimensions ---*/
  int dim_x = dims.ncell_x;
  int dim_y = dims.ncell_y;
  int dim_z = dims.ncell_z;

  /*--- Array Sizes ---*/
  int facexy_size = dims.ncell_x * dims.ncell_y * NE * NA * NU * NOCTANT;
  int facexz_size = dims.ncell_x * dims.ncell_z * NE * NA * NU * NOCTANT;
  int faceyz_size = dims.ncell_y * dims.ncell_z * NE * NA * NU * NOCTANT;
  int local_size = dims.ncell_x * dims.ncell_y * NE * NA * NU * NOCTANT;
  int v_size = NM * NA * NOCTANT;
  int input_size = dims.ncell_x * dims.ncell_y * dims.ncell_z * NE * NU * NM;
  int output_size = dims.ncell_x * dims.ncell_y * dims.ncell_z * NE * NU * NM;

  /*--- KBA sweep ---*/

#pragma acc data present(facexy[:facexy_size], \
			 facexz[:facexz_size], \
			 faceyz[:faceyz_size], \
                         m_from_a[:v_size], \
			 local[:local_size], \
			 output[:output_size], dims, this)
  {

/*---Loop over octants---*/
for( octant=0; octant<NOCTANT; ++octant )
  {

  /*--- Number of wavefronts equals the sum of the dimension sizes
    minus the number of dimensions minus one. In our case, we have
    three total dimensions, so we add the sizes and subtract 2. 
  ---*/
  int num_wavefronts = (dim_z + dim_y + dim_x) - 2;

  for (wavefront=0; wavefront < num_wavefronts; wavefront++)
    {

#pragma acc parallel async(octant)
{
#pragma acc loop independent gang, collapse(2)
    for (iy=0; iy<dim_y; ++iy)
      for (ix=0; ix<dim_x; ++ix)
	{

	  /*--- Solve for Z and check bounds ---*/
	  iz = wavefront - (ix + iy);

	  /*--- Bounds check ---*/
	  if (iz >= 0 && iz < dim_z)
	    {
	      solve(ix, iy, iz, octant);
	      
	    } /*--- iz ---*/
	} /*--- iy,ix ---*/
 } /*--- #pragma acc parallel ---*/
    } /*--- wavefront ---*/
  } /*--- octant ---*/
  }   /*--- #pragma acc data present ---*/

#pragma acc wait

  /*--- Copy output from the GPU and free GPU memory ---*/
#pragma acc exit data copyout(output[:output_size]), delete(facexy[:facexy_size], \
							    facexz[:facexz_size], \
							    faceyz[:faceyz_size], \
                                                             m_from_a[:v_size], \
			                                    local[:local_size], \
							    input[:input_size])
}

/*--- Print Output ---*/
void RT::print(){
  int ie = 0;
  int iu = 0;
  int im = 0;
  for (int iz=0; iz<dims.ncell_z; iz++)
    for (int iy=0; iy<dims.ncell_y; iy++)
      for (int ix=0; ix<dims.ncell_x; ix++)
	{
	  printf("output[%d][%d][%d] = %f\n", iz, iy, ix,
		 output[im + NM * ( 
			iu + NU * (
			ie + NE * (
		        ix + dims.ncell_x * (
                        iy + dims.ncell_y * (
	                iz + dims.ncell_z * (
					     0 )))))) ]
);
	}
}

/*--- Solver ---*/
/*--- (This is called for each gridcell). This will contain your algorithm. ---*/
#pragma acc routine vector
void RT::solve(int ix,
	       int iy,
	       int iz,
	       int octant) {

  /*--- Declarations ---*/
  int ia = 0;
  int iu = 0;
  int ie = 0;
  int im = 0;
  int dim_ne = NE;
  int dim_nu = NU;
  int dim_na = NA;
  int dim_nm = NM;

  /*--- In-Gridcell Computations ---*/

#pragma acc loop independent vector, collapse(3)
  for (ie=0; ie<dim_ne; ++ie)
  for( iu=0; iu<dim_nu; ++iu )
  for( ia=0; ia<dim_na; ++ia ) 
    {
      real result = (real)0.0;

#pragma acc loop seq
      for ( im=0; im<dim_nm; ++im) {

	real a_from_m = 1.0/im;

      /*--- Input read ---*/
	result += input[im + NM * ( 
                        iu + NU * (
 	                ie + NE * (
		  	ix + dims.ncell_x * (
		  	iy + dims.ncell_y * (
		  	iz + dims.ncell_z * (
		  
	  0 )))))) ] * 
	  m_from_a[im + NM * (
	  	   ia + NA * (
	  	   octant + NOCTANT * (
	  			       0)))];

      } /*--- NM ---*/

      local[ia + NA * (
	    iu + NU * (
	    ie + NE * (
            ix + dims.ncell_x * (
            iy + dims.ncell_y * (
	    octant + NOCTANT * (
				0 )))))) ] = result;
    } /*--- NE ---*/

#pragma acc loop independent vector, collapse(2)
  for( ie=0; ie<dim_ne; ++ie )
  for( ia=0; ia<dim_na; ++ia )
    {
      compute(ix, iy, iz, ie, ia, octant);
    }

  /*--- Transform and write ---*/
#pragma acc loop independent vector, collapse(3)
  for (ie=0; ie<dim_ne; ++ie)
  for( iu=0; iu<dim_nu; ++iu )
  for ( im=0; im<dim_nm; ++im)
    {
      real result = (real)0;
#pragma acc loop seq
      for( ia=0; ia<dim_na; ++ia )
      {
	result += local[ia + NA * (
			iu + NU * (
			ie + NE * (
          	    	ix + dims.ncell_x * (
                        iy + dims.ncell_y * (
			octant + NOCTANT * (
					    0 )))))) ] * 
	  m_from_a[im + NM * (
	  	   ia + NA * (
	  	   octant + NOCTANT * (
	  			       0)))];
      }

    #pragma acc atomic update
	output[im + NM * (
	       iu + NU * (
	       ie + NE * (
	       ix + dims.ncell_x * (
               iy + dims.ncell_y * (
	       iz + dims.ncell_z * (
				    0 )))))) ] += result;
  } /*--- ie ---*/
}

/*=============================================*/
/*=== ALGORITHM-SPECIFIC FUNCTIONS ===*/
/*=============================================*/

/*===========================================================================*/
/*--- compute routine ---*/

#pragma acc routine seq
  void RT::compute(int ix, int iy, int iz, int ie, int ia, int octant){
  /*---Average the face values and accumulate---*/

  /*---The state value and incoming face values are first adjusted to
    normalized values by removing the spatial scaling.
    They are then combined using a weighted average chosen in a special
    way to give just the expected result.
    Finally, spatial scaling is applied to the result which is then
    stored.
    ---*/

    int iu = 0;

  /*--- Quantities_scalefactor_octant_ inline ---*/
  const real scalefactor_octant = (real)1.0 + octant;
  const real scalefactor_octant_r = ((real)1) / scalefactor_octant;

  /*---Quantities_scalefactor_space_ inline ---*/
  const real scalefactor_space = (real)Quantities_scalefactor_space(ix, iy, iz);
  const real scalefactor_space_r = ((real)1) / scalefactor_space;
  const real scalefactor_space_x_r = ((real)1) /
    Quantities_scalefactor_space( ix - 1, iy, iz );
  const real scalefactor_space_y_r = ((real)1) /
    Quantities_scalefactor_space( ix, iy - 1, iz );
  const real scalefactor_space_z_r = ((real)1) /
    Quantities_scalefactor_space( ix, iy, iz - 1 );

#pragma acc loop seq
  for( iu=0; iu<NU; ++iu )
    {

      int local_index = ia + NA * (
			iu + NU * (
			ie + NE * (
		        ix + dims.ncell_x * (
		        iy + dims.ncell_y * (
			octant + NOCTANT * (
					    0))))));

  const real result = (real)1.0/( local[local_index] * scalefactor_space_r + 
           (
	    /*--- ref_facexy inline ---*/
	    facexy[ia + NA * (
		   iu + NU * (
                   ie + NE * (
                   ix + dims.ncell_x * (
                   iy + dims.ncell_y * (
		   octant + NOCTANT * (
				       0 )))))) ]

	    /*--- Quantities_xfluxweight_ inline ---*/
	    * (real) ( 1 / (real) 2 )
	    
	    * scalefactor_space_z_r
	    
	    /*--- ref_facexz inline ---*/
	    + facexz[ia + NA * (
		     iu + NU * (
                     ie + NE * (
                     ix + dims.ncell_x * (
                     iz + dims.ncell_z * (
		     octant + NOCTANT * (
					 0 )))))) ]

	    /*--- Quantities_yfluxweight_ inline ---*/
	    * (real) ( 1 / (real) 4 )

	    * scalefactor_space_y_r
	    
	    /*--- ref_faceyz inline ---*/
	    + faceyz[ia + NA * (
		     iu + NU * (
                     ie + NE * (
                     iy + dims.ncell_y * (
                     iz + dims.ncell_z * (
		     octant + NOCTANT * (
					 0 )))))) ]

	    /*--- Quantities_zfluxweight_ inline ---*/
	    * (real) ( 1 / (real) 4 - 1 / (real) (1 << ( ia & ( (1<<3) - 1 ) )) )

	    * scalefactor_space_x_r
	    ) 
	    * scalefactor_octant_r ) * scalefactor_space;

  local[local_index] = result;

  const real result_scaled = result * scalefactor_octant;
  /*--- ref_facexy inline ---*/
  facexy[ia + NA * (
	 iu + NU * (
         ie + NE * (
         ix + dims.ncell_x * (
         iy + dims.ncell_y * (
	 octant + NOCTANT * (
			     0 )))))) ] = result_scaled;

  /*--- ref_facexz inline ---*/
  facexz[ia + NA * (
	 iu + NU * (
         ie + NE * (
         ix + dims.ncell_x * (
         iz + dims.ncell_z * (
	 octant + NOCTANT * (
			     0 )))))) ] = result_scaled;

  /*--- ref_faceyz inline ---*/
  faceyz[ia + NA * (
	 iu + NU * (
         ie + NE * (
         iy + dims.ncell_y * (
         iz + dims.ncell_z * (
	 octant + NOCTANT * (
			     0 )))))) ] = result_scaled;

    } /*---for---*/
}

/*===========================================================================*/
/*--- Quantities_init_face routine ---*/

#pragma acc routine seq
real RT::Quantities_init_face(int ia, int ie, int iu, int scalefactor_space, int octant)
{
  /*--- Quantities_init_facexy inline ---*/

  /*--- Quantities_affinefunction_ inline ---*/
  return ( (real) (1 + ia) ) 

    /*--- Quantities_scalefactor_angle_ inline ---*/
    * ( (real) (1 << (ia & ( (1<<3) - 1))) ) 

    /*--- Quantities_scalefactor_space_ inline ---*/
    * ( (real) scalefactor_space)

    /*--- Quantities_scalefactor_energy_ inline ---*/
    * ( (real) (1 << ((( (ie) * 1366 + 150889) % 714025) & ( (1<<2) - 1))) )

    /*--- Quantities_scalefactor_unknown_ inline ---*/
    * ( (real) (1 << ((( (iu) * 741 + 60037) % 312500) & ( (1<<2) - 1))) )

    /*--- Quantities_scalefactor_octant_ ---*/
    * ( (real) 1 + octant);
}

/*===========================================================================*/
/*--- Quantities_scalefactor_space routine ---*/

#pragma acc routine seq
int RT::Quantities_scalefactor_space(int ix, int iy, int iz)
{
  /*--- Quantities_scalefactor_space ---*/
  int scalefactor_space = 0;

  scalefactor_space = ( (scalefactor_space+(ix+2))*8121 + 28411 ) % 134456;
  scalefactor_space = ( (scalefactor_space+(iy+2))*8121 + 28411 ) % 134456;
  scalefactor_space = ( (scalefactor_space+(iz+2))*8121 + 28411 ) % 134456;
  scalefactor_space = ( (scalefactor_space+(ix+3*iy+7*iz+2))*8121 + 28411 ) % 134456;
  scalefactor_space = ix+3*iy+7*iz+2;
  scalefactor_space = scalefactor_space & ( (1<<2) - 1 );
  scalefactor_space = 1 << scalefactor_space;

  return scalefactor_space;
}

/*===========================================================================*/
/*--- Quantities_scalefactor_space routine ---*/

#pragma acc routine seq
int RT::Quantities_scalefactor_energy(int ie)
{
  const int im = 714025;
  const int ia = 1366;
  const int ic = 150889;

  int result = ( (ie)*ia + ic ) % im;
  result = result & ( (1<<2) - 1 );
  result = 1 << result;

  return result;
}
