#include "range.hpp"
#define BLOCK_SIZE 256
#define TILE_WIDTH 16

__constant__ int xdims_cd[4];
__constant__ int wdims_cd[4];
__constant__ int ydims_cd[4];

__global__ void matrixMultiplyShared(const float *A, const float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns,
                                     int coffset, unsigned int idx);

 __global__ void matrixMultiply_t(const float *A, const float *B, float *C, int numARows,
		                              int numAColumns, int numBRows, int numBColumns, int numCRows, 
                                  int numCColumns, int coffset, unsigned int idx);

__global__ void matrixMultiply_t(const float *A, const float *B, float *C, int numARows,
		                             int numAColumns, int numBRows,
	                               int numBColumns, int numCRows, int numCColumns, int coffset, unsigned int idx);

__global__ void x_transform_kernel(const float *X,float * X_t);

__global__ void unroll_kernel(const float *X, float *X_unroll, const int xoffset, unsigned int idx );

__global__ void relu2_Kernel( float* X , int length );

__global__ void average_pool_kernel(const float *X, const int *xdims,
                                    const int pool_size, float *Y, const int *ydims);

__global__ void FullyForward_Kernel( float* X , float* W, float *Y, int sz , int xdim, int wdim );

