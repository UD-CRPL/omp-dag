/*---------------------------------------------------------------------------*/
/*!
 * \file   gauss_seidel.cpp
 * \author Robert Searles
 * \date   Wed Jan 2 16:52:00 EST 2019
 * \brief  Gauss Seidel in-gridcell algorithm
 * \note   
 */
/*---------------------------------------------------------------------------*/

#include "dimensions.h"
#include "gauss_seidel.h"
#include <stdio.h>

/*=============================================*/
/*=== MANDATORY FUNCTIONS ===*/
/*=============================================*/

/*--- Constructor ---*/
GS::GS(Dimensions d) {
	printf("Creating Gauss Seidel Simulation\n\n");

	dims = d;

	/*--- Dimensions ---*/
	ncellx = dims.ncell_x;
	ncelly = dims.ncell_y;
	nu = dims.ncomponents;

	/*---Sizes---*/
	v_size = nu*ncellx*ncelly;
	a_size = nu*nu*ncellx*ncelly;

	/*--- Allocate Arrays ---*/
	v1 = (real*) malloc(v_size*sizeof(real));
	v2 = (real*) malloc(v_size*sizeof(real));
	an = (real*) malloc(a_size*sizeof(real));
	as = (real*) malloc(a_size*sizeof(real));
	ae = (real*) malloc(a_size*sizeof(real));
	aw = (real*) malloc(a_size*sizeof(real));
}

/*--- Initialization ---*/
void GS::init() {
	int iy = 0;
	int ix = 0;
	int iu = 0;
	int ju = 0;

	for (iy=1; iy<ncelly-1; ++iy)
	{
		for (ix=1; ix<ncellx-1; ++ix)
		{

			for (iu=0; iu<nu; ++iu)
			{

				for (ju=0; ju<nu; ++ju)
				{
					an[ju+nu*(iu+nu*(ix+ncellx*iy))] = 0.0;
					as[ju+nu*(iu+nu*(ix+ncellx*iy))] = 0.0;
					ae[ju+nu*(iu+nu*(ix+ncellx*iy))] = 0.0;
					aw[ju+nu*(iu+nu*(ix+ncellx*iy))] = 0.0;
				}
				/*---Initial guess is 1 on interior, 0 on bdry---*/
				/*---Assume right hand side is 0---*/
				v1[iu+nu*(ix+ncellx*iy)] = 1.0;
				v2[iu+nu*(ix+ncellx*iy)] = 1.0;
				/*---Constant coefficient Laplacian operator---*/
				an[iu+nu*(iu+nu*(ix+ncellx*iy))] = 0.25;
				as[iu+nu*(iu+nu*(ix+ncellx*iy))] = 0.25;
				ae[iu+nu*(iu+nu*(ix+ncellx*iy))] = 0.25;
				aw[iu+nu*(iu+nu*(ix+ncellx*iy))] = 0.25;
			}
		}
	}

	/*---Initializations: boundary---*/
	/*---Assume Dirichlet zero boundaries---*/


	for (iy=0; iy<ncelly; ++iy)
	{

		for (iu=0; iu<nu; ++iu)
		{
			v1[iu+nu*(0+ncellx*iy)] = 0.0;
			v2[iu+nu*(0+ncellx*iy)] = 0.0;
			v1[iu+nu*((ncellx-1)+ncellx*iy)] = 0.0;
			v2[iu+nu*((ncellx-1)+ncellx*iy)] = 0.0;
		}
	}


	for (ix=0; ix<ncellx; ++ix)
	{

		for (iu=0; iu<nu; ++iu)
		{
			v1[iu+nu*(ix+ncellx*0)] = 0.0;
			v2[iu+nu*(ix+ncellx*0)] = 0.0;
			v1[iu+nu*(ix+ncellx*(ncelly-1))] = 0.0;
			v2[iu+nu*(ix+ncellx*(ncelly-1))] = 0.0;
		}
	}
} /*--- GS::init() ---*/

/*--- Run ---*/
/*--- This should run your wavefront sweep component & in-gridcell computation ---*/
void GS::run(){
	int iteration;
	for (iteration=0; iteration<dims.niterations; ++iteration) {
		const real* const __restrict__ vi = iteration%2 ? v1 : v2;
		real* const __restrict__ vo = iteration%2 ? v2 : v1;

		/*--- Solver ---*/
		GS::solve(nu, ncellx, ncelly, vo, vi, an, as, ae, aw);
	}
	/*--- Results ---*/
	vfinal = dims.niterations%2 ? v1 : v2;

	/*---Deallocations---*/
	free(an);
	free(as);
	free(ae);
	free(aw);

} /*--- GS::run() ---*/

/*--- Print Output ---*/
void GS::print(){
	int iy,ix,iu;

	for (iy=0; iy<ncelly; ++iy)
		for (ix=0; ix<ncellx; ++ix)
			printf("vfinal[%d][%d][%d] = %f\n", iy, ix, 0, vfinal[0+nu*(ix+ncellx*iy)]);

	/*---Deallocations---*/
	free(v1);
	free(v2);
}

/*=============================================*/
/*=== ALGORITHM-SPECIFIC FUNCTIONS ===*/
/*=============================================*/

/*--- Solver ---*/
void GS::solve(int nu, int ncellx, int ncelly, 
		real* __restrict__ vo,
		const real* __restrict__ vi,
		const real* __restrict__ an,
		const real* __restrict__ as,
		const real* __restrict__ ae,
		const real* __restrict__ aw)
{
	const int nwavefronts = (ncellx + ncelly) - 1;
	int wavefront = 0;

	/*---Loop over wavefronts---*/
	for (wavefront=0; wavefront<nwavefronts; ++wavefront)
	{
		int ix;
		/*---Loop over gridcells in wavefront---*/

		for (ix=1; ix<ncellx-1; ++ix)
		{
			const int iy = wavefront - ix;
			if (iy > 1 && iy < ncelly-1)
				process_cell(ix, iy, nu, ncellx, ncelly, vo, vi, an, as, ae, aw);
		}
	}
} /*--- GS::solve() ---*/

/*===========================================================================*/
/*---5-point stencil operation applied at a gridcell---*/


void GS::process_cell(int ix, int iy, int nu, int ncellx, int ncelly,
		real* __restrict__ vo,
		const real* __restrict__ vi,
		const real* __restrict__ an,
		const real* __restrict__ as,
		const real* __restrict__ ae,
		const real* __restrict__ aw)
{
	int iu = 0;
	int ju = 0;

	/*---South, east connections use "old" values, north, west use "new"---*/
	/*---The connection operation is matrix-vector product---*/


	for (iu=0; iu<nu; ++iu)
	{

		real sum = 0.0;
		vo[iu+nu*(ix+ncellx*iy)] = 0.0;


		for (ju=0; ju<nu; ++ju)
		{
			sum +=

					aw[ju+nu*(iu+nu*(ix+ncellx*iy))] *
					vo[iu+nu*((ix-1)+ncellx*iy)] +

					an[ju+nu*(iu+nu*(ix+ncellx*iy))] *
					vo[iu+nu*(ix+ncellx*(iy-1))] +

					ae[ju+nu*(iu+nu*(ix+ncellx*iy))] *
					vi[iu+nu*((ix+1)+ncellx*iy)] +

					as[ju+nu*(iu+nu*(ix+ncellx*iy))] *
					vi[iu+nu*(ix+ncellx*(iy-1))];
		}

		vo[iu+nu*(ix+ncellx*iy)] = sum;

	}
} /*--- GS::process_cell() ---*/
