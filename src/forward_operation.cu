#include "forward_operation.hpp"


// Matrix multiplication with shared memory (does not improve performance)
__global__ void matrixMultiplyShared(const float *A, const float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns,
                                     int coffset, unsigned int idx) {
  const int tile_w = 16;
  extern __shared__ float shmem[];
  float * sub_A = &shmem[0];
  float * sub_B = &shmem[tile_w*tile_w];
  
  int tx=threadIdx.x;
  int ty=threadIdx.y; 
  int bx=blockIdx.x;
  int by=blockIdx.y;
  
  int row = by * tile_w + ty;
  int col = bx * tile_w + tx;
    
  float Cvalue=0.0;
  for (int m=0;m<(numAColumns+tile_w-1)/tile_w;++m){
    if (m*tile_w + tx < numAColumns && row < numARows )
      sub_A[ty*tile_w+tx] = A[row*numAColumns + m*tile_w+tx];
    else 
      sub_A[ty*tile_w+tx] =0.0;
    if(m*tile_w + ty < numBRows && col < numBColumns ){
      sub_B[ty*tile_w+tx] = B[ idx * numBColumns* numBRows +(m*tile_w+ty)*numBColumns+col];  
    }
    else 
      sub_B[ty*tile_w+tx] =0.0;
    __syncthreads();
    for (int k=0;k<tile_w;++k){
      Cvalue+=sub_A[ty*tile_w+k]*sub_B[k*tile_w+tx];
    }
    __syncthreads();
  }
  if(row<numCRows && col < numCColumns){
    C[coffset + row*numCColumns+col]=Cvalue;
  }
}

__global__ void matrixMultiply_t(const float *A, const float *B, float *C, int numARows,
		               int numAColumns, int numBRows,
	                       int numBColumns, int numCRows, int numCColumns, int coffset, unsigned int idx) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if ((row < numCRows && col < numCColumns)){
    float Pvalue = 0;
    for(int k = 0; k < numAColumns; ++k){
      Pvalue += A[ k * numARows + row ] * B[idx * numBColumns* numBRows  + k * numBColumns + col ];
    }
    C[col * numCRows + row + coffset] = Pvalue;
  }
}

__global__ void x_transform_kernel(const float *X,float * X_t){
  unsigned int i = blockIdx.x;
  unsigned int c = blockIdx.y; 
  unsigned int h = threadIdx.x;
  unsigned int w = threadIdx.y;

  unsigned int xoff = i* xdims_cd[1]*xdims_cd[2]*xdims_cd[3]+ 
                      h* xdims_cd[2]*xdims_cd[3]+
                      w *xdims_cd[3] +
                      c;
  unsigned int xoff_t = i* xdims_cd[3]*xdims_cd[1]*xdims_cd[2]+ 
                        c* xdims_cd[1]*xdims_cd[2]+
                        h *xdims_cd[2]+ 
                        w;
  X_t[xoff_t]=X[xoff];
}

__global__ void unroll_kernel(const float *X, float *X_unroll, const int xoffset, unsigned int idx ){
  int t = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  const auto H_filter = wdims_cd[0];
  const auto W_filter = wdims_cd[1];
  const auto in_channel = wdims_cd[2];
  const auto H_X = xdims_cd[1];
  const auto W_X = xdims_cd[2];
  const auto H_out = H_X - H_filter + 1;
  const auto W_out = W_X - W_filter + 1;
  const auto W_unroll = H_out * W_out;
  if(t < in_channel * W_unroll){
    const int c = t / W_unroll;
    const int s = t % W_unroll;
    const int row_out = s / W_out;
    const int col_out = s % W_out;
    const int col_unroll = row_out * W_out + col_out;
    const int row_base = c * H_filter * W_filter;
    const auto size = in_channel * H_filter * W_filter * W_out * H_out ;
    const auto x_unroll_off = col_unroll + idx *size + row_base* W_unroll; 
    const auto x_offset_offset =  xoffset+ c * (H_X * W_X) + row_out * (W_X) + col_out ;
    for(const auto p : range(0, H_filter)){
      for(const auto q : range(0, W_filter)){
        X_unroll[x_unroll_off + (p * H_filter + q) * W_unroll] = X[x_offset_offset +p* W_X + q];
	    }
    }
  }
}


__global__ void relu2_Kernel( float* X , int length ){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if( i < length ){
    X[i] = (X[i] < 0) ? 0 : X[i];
  }
}

__global__ void average_pool_kernel(const float *X, const int *xdims,
                         const int pool_size, float *Y, const int *ydims) {
  int batch = blockIdx.x;
  int map = blockIdx.y;
  int col_o = threadIdx.y;
  int row_o = threadIdx.x;
  float sum = 0;
  int n_Y_elements = ydims[0] * ydims[1] * ydims[2] * ydims[3];
  int n_X_elements = xdims[0] * xdims[1] * xdims[2] * xdims[3];
  for(auto row_i = row_o * pool_size; row_i < ((row_o + 1) * pool_size); ++row_i){
    for(auto col_i = col_o * pool_size; col_i < ((col_o + 1) * pool_size); ++col_i){
      const auto xoffset = batch * xdims[1] * xdims[2] * xdims[3] 
                         + row_i * xdims[2] * xdims[3] 
                         + col_i * xdims[3]
                         + map;
      if(xoffset < n_X_elements){
        if(X[xoffset]>0){
          sum += X[xoffset];
        }
      }
      else
      {
        // This is for debugging 
        printf("xoffset = %d, n_X_elements = %d, batch = %d, map = %d, col_o = % d, row_o = %d, col_i = % d, row_i = %d\n", xoffset, n_X_elements, batch, map, col_o, row_o, col_i, row_i);
      } 
    }
  }
  const auto yoffset = batch * ydims[1] * ydims[2] * ydims[3] 
                         + row_o * ydims[2] * ydims[3] 
                         + col_o * ydims[3]
                         + map;
  if(yoffset < n_Y_elements){
    Y[yoffset] = sum / (pool_size * pool_size);
  }
  else{
    printf("yoffset = %d, n_Y_elements = %d", yoffset, n_Y_elements);
  }
  
}



__global__ void FullyForward_Kernel( float* X , float* W, float *Y, int sz , int xdim, int wdim ){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if( row < wdim && col < xdim  ){
    float sum = 0.0;
    for (const auto k : range(0, sz)) {
      sum += X[col * sz + k] * W[k * wdim + row];
    }
    Y[col * wdim + row] = sum;
  }
}
