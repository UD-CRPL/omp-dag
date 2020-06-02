/*---------------------------------------------------------------------------*/
/*!
 * \file   local_sequence_alignment.cpp
 * \author Robert Searles
 * \date   Tue Sep 19 15:46:00 EST 2018
 * \brief  Local Sequence Alignment in-gridcell algorithm
 * \note   
 */
/*---------------------------------------------------------------------------*/

#include "dimensions.h"
#include "local_sequence_alignment.h"
#include <stdio.h>
#include <algorithm>
#include <cstdlib>
#include <string>

/*=============================================*/
/*=== MANDATORY FUNCTIONS ===*/
/*=============================================*/

/*--- Constructor ---*/
LSA::LSA(Dimensions d) {
	printf("Creating Local Sequence Alignment Simulation\n\n");
	dims=d;
	if(dims.ncomponents==32) {
		seq1=seq1_cstr;
		seq2=seq2_cstr;
	}
	else {
		std::string seq1_str=generate_sequence(dims.ncell_x,"ATCG");
		std::string seq2_str=generate_sequence(dims.ncell_y,"ATCG");
		seq1 = new char [seq1_str.length()+1];
		strcpy(seq1, seq1_str.c_str());
		seq2 = new char [seq2_str.length()+1];
		strcpy(seq2, seq2_str.c_str());
	}
	rows = strlen(seq1) + 1;
	cols = strlen(seq2) + 1;

	matrix_size = rows*cols;

#pragma acc enter data copyin(seq1[:rows-1], seq2[cols-1], this)

	/*--- Array Allocation ---*/
	matrix = (int*)malloc_host_int(matrix_size);
#pragma acc enter data create(matrix[:matrix_size])
}
LSA::~LSA(){
	if(seq1!=nullptr)
		delete [] seq1;
	seq1=nullptr;
	if(seq2!=nullptr)
		delete [] seq2;
	seq2=nullptr;
}

int LSA::r(){
	return rows;
}
int LSA::c(){
	return cols;
}
/*--- Initialization ---*/
void LSA::init(){
#pragma acc parallel present(matrix[:matrix_size], this)
	{
#pragma acc loop independent gang
		for (int x = 0; x < rows; x++)
#pragma acc loop independent vector
			for (int y = 0; y < cols; y++)
				matrix[(x * cols) + y] = 0;
	} /*--- #pragma acc parallel ---*/
}

/*--- Run ---*/
/*--- This should run your wavefront sweep component & in-gridcell computation ---*/
void LSA::run(){
	create_score_matrix(rows, cols);
}

/*--- Print Output ---*/
void LSA::print(){
	for (int x = 1; x < rows; x++)
		//    for (int y = 1; y < cols; y++)
		printf("matrix[%d][%d] = %d\n", x, cols-1, matrix[(x * cols) + cols-1]);
}

/*=============================================*/
/*=== ALGORITHM-SPECIFIC FUNCTIONS ===*/
/*=============================================*/

/*--- Create sequence alignment scoring matrix ---*/
void LSA::create_score_matrix(int rows, int cols)
{
	/*--- KBA sweep ---*/
#pragma acc data present(matrix[:matrix_size], seq1[:rows-1], seq2[:cols-1], this)
	{
		int max_score = 0;
		int wavefront = 0;

		/*--- Number of wavefronts equals the sum of the dimension sizes
    minus the number of dimensions minus one. In our case, we have
    three total dimensions, so we add the sizes and subtract 2. 
  ---*/
		int num_wavefronts = (rows + cols) - 1;

		for (wavefront=0; wavefront < num_wavefronts; wavefront++)
		{

#pragma acc parallel //reduction(max:max_score)
			{
				// for (int x = 1; x < rows; x++)
#pragma acc loop gang vector
				for (int y = 1; y < cols; y++)
				{
					/*--- Solve for X and check bounds ---*/
					int x = wavefront - y;

					/*--- Bounds check ---*/
					if (x >= 1 && x < rows)
						// for (int x=1; x<rows; x++)
						//   for (int y=1; y<cols; y++)
					{
						//int score = calc_score(x, y);

						// if (score > max_score)
						//   max_score = score;

						matrix[(x * cols) + y] = calc_score(x, y);
					} /*--- x ---*/
				} /*--- for y ---*/
			} /*--- #pragma acc parallel ---*/
		} /*--- wavefront ---*/
	} /*--- #pragma acc data ---*/

#pragma acc wait
#pragma acc exit data copyout(matrix[:matrix_size]), \
		delete(seq1[:rows-1], seq2[:cols-1])
}

/*--- Calculate gridcell score ---*/
#pragma acc routine seq
int LSA::calc_score(int x, int y)
{
	int similarity;
	if (seq1[x - 1] == seq2[y - 1])
		similarity = match;
	else
		similarity = mismatch;

	/*--- Calculate scores ---*/
	int diag_score = matrix[(x - 1) * cols + (y - 1)] + similarity;
	int up_score   = matrix[(x - 1) * cols + y] + gap;
	int left_score = matrix[x * cols + (y - 1)] + gap;

	/*--- Take max of scores and return ---*/
	int result = 0;
	if (diag_score > result)
		result = diag_score;
	if (up_score > result)
		result = up_score;
	if (left_score > result)
		result = left_score;

	return result;
}
