#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <fstream>
#include <numeric>
#include <map>
#include <sys/time.h>
#include <valarray>
#include <iomanip>
#include <hdf5.h>
#include "cublas_v2.h"
#include "range.hpp"
#include "utils.hpp"
#include "forward_operation.hpp"

#define NUM_ROWS 28
#define NUM_COLS 28
#define NUM_CHANNELS 1
#define NUM_DIGITS 10

static int FLAGS_batch_size = 10000;
static std::string FLAGS_testdata{};
static std::string FLAGS_model{};

// Data and reference data dimensions
static int xdims[] = {FLAGS_batch_size, NUM_ROWS, NUM_COLS, NUM_CHANNELS};
static int rdims[] = {FLAGS_batch_size, NUM_DIGITS};

// Model dimensions
static int conv1dims[] = {5, 5, 1, 32};
static int conv2dims[] = {5, 5, 32, 64};
static int fc1dims[]   = {1024, 128};
static int fc2dims[]   = {128, 10};

static int loadData(float *x, float *y) {
  // Open the data file
  const auto file_id =
      H5Fopen(FLAGS_testdata.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

  // Open the dataset x and y
  const auto x_id = H5Dopen2(file_id, "/x", H5P_DEFAULT);
  const auto y_id = H5Dopen2(file_id, "/y", H5P_DEFAULT);

  // Get the dataset x dimensions
  const auto xspace = H5Dget_space(x_id);
  const auto xndims = H5Sget_simple_extent_ndims(xspace);
  assert(xndims == 4);

  hsize_t input_dims[xndims];
  H5Sget_simple_extent_dims(xspace, input_dims, NULL);
  if (input_dims[0] != FLAGS_batch_size) {
    std::cout << "data size does not match batch size specified!\n";
    return 1; // return error
  }
  std::cout << "input dimensions = " << input_dims[0] << " x " << input_dims[1]
            << " x " << input_dims[2] << " x " << input_dims[3] << "\n";

  // Read the dataset x and y
  check_success(
      H5Dread(x_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, x));
  check_success(
      H5Dread(y_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, y));

  // Close the dataset x and y
  check_success(H5Dclose(x_id));
  check_success(H5Dclose(y_id));

  // Close the file
  check_success(H5Fclose(file_id));

  // return success
  return 0;
}

static void loadModel(float *conv1, float *conv2, float *fc1, float *fc2) {
  // Open the model file
  const auto file_id = H5Fopen(FLAGS_model.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

  // Open the dataset
  const auto conv1_id = H5Dopen2(file_id, "/conv1", H5P_DEFAULT);
  const auto conv2_id = H5Dopen2(file_id, "/conv2", H5P_DEFAULT);
  const auto fc1_id   = H5Dopen2(file_id, "/fc1", H5P_DEFAULT);
  const auto fc2_id   = H5Dopen2(file_id, "/fc2", H5P_DEFAULT);

  // Read the dataset
  check_success(H5Dread(conv1_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, conv1));
  check_success(H5Dread(conv2_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, conv2));
  check_success(
      H5Dread(fc1_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fc1));
  check_success(
      H5Dread(fc2_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fc2));

  // Close the dataset x and y
  check_success(H5Dclose(conv1_id));
  check_success(H5Dclose(conv2_id));
  check_success(H5Dclose(fc1_id));
  check_success(H5Dclose(fc2_id));

  // Close the file
  check_success(H5Fclose(file_id));
}


static void x_transform(const int xdims[4], const float *d_X,float * d_Xt){

  dim3 grid_dim(xdims[0],xdims[3],1); 
  dim3 block_dim(xdims[2], xdims[1],1);
  
  x_transform_kernel<<<grid_dim,block_dim>>>(d_X, d_Xt);

  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess)
    printf("kernel launch failed with error \"%s\".\n",  cudaGetErrorString(cudaerr));

}





cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %sn", 
            cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}



static void mmul_gpu(const float *d_A, const float *d_B, float *d_C,
              int numARows, int numAColumns,
	      int numBRows, int numBColumns, int numCRows, int numCColumns, int coffset,cudaStream_t & stream, unsigned int idx){
  const int tileWidth = 16;
  int maxdim = numCRows > numCColumns ? numCRows : numCColumns;
  dim3 dimGrid(maxdim / tileWidth, maxdim / tileWidth, 1);
  if (numCColumns % tileWidth) dimGrid.x ++;
  if (numCRows % tileWidth) dimGrid.y ++;
  dim3 dimBlock(tileWidth, tileWidth, 1);
  //size_t shmem_size = sizeof(float) * tileWidth*tileWidth*2;
  //matrixMultiplyShared<<<dimGrid, dimBlock,shmem_size, stream>>>(d_A, d_B, d_C, numARows, numAColumns,
  //                     numBRows, numBColumns, numCRows, numCColumns, coffset, idx);
  matrixMultiply_t<<<dimGrid, dimBlock, 0, stream>>>(d_A, d_B, d_C, numARows, numAColumns, 
					      numBRows, numBColumns, numCRows, numCColumns, coffset, idx);
}



static void unroll_gpu(const float* d_X, float *d_X_unrolled, const int in_channel, const int H_out, const int W_out, const int xoffset, 
            cudaStream_t& stream, unsigned int idx){
  dim3 dim_grid((in_channel * H_out * W_out) / BLOCK_SIZE, 1, 1);
  if((in_channel * H_out * W_out) % BLOCK_SIZE) ++dim_grid.x;
  dim3 dim_block(BLOCK_SIZE, 1, 1);
  unroll_kernel<<<dim_grid, dim_block, 0, stream>>>(d_X, d_X_unrolled, xoffset, idx); 

}





// Convolution Layer (reduce to matrix multiplication)
static void conv_mat(const float *d_X, const int xdims[4],
                    const float *d_W, const int wdims[4], float *d_Y,
                    const int ydims[4]){

  const auto start = now();

  cudaMemcpyToSymbol(xdims_cd, xdims, 4 * sizeof(int));
  cudaMemcpyToSymbol(wdims_cd, wdims, 4 * sizeof(int));
  cudaMemcpyToSymbol(ydims_cd, ydims, 4 * sizeof(int));

  const auto H_filter = wdims[0];
  const auto W_filter = wdims[1];
  const auto in_channel = wdims[2];
  const auto out_channel = wdims[3];
  const auto batch_size = xdims[0];
  const auto x_h = xdims[1];
  const auto x_w = xdims[2];
  const auto H_out = x_h - H_filter + 1;
  const auto W_out = x_w - W_filter + 1;
  const auto W_unroll = H_out * W_out;
  const auto H_unroll = in_channel * H_filter * W_filter;
  float *d_X_unrolled;
  const auto stream_size=16;
  cudaMalloc((void **)&d_X_unrolled, in_channel * H_filter * W_filter * W_out * H_out* stream_size * sizeof(float));
  cudaStream_t stream[stream_size];


  // Create Stream
  for( auto i :range(0,stream_size) )
      cudaStreamCreate( &stream[i] );

  // Each iteration a unroll & a matrix multiplication are put into a stream (based on index modulus)
  for(const auto i : range(0, batch_size)){
    const auto xoffset = i * xdims[1] * xdims[2] * xdims[3];
    const auto yoffset = i * ydims[1] * ydims[2] * ydims[3];

    unroll_gpu(d_X, d_X_unrolled, in_channel, H_out, W_out, xoffset, stream[i%stream_size], i%stream_size);
    mmul_gpu(d_W, d_X_unrolled, d_Y, out_channel, H_unroll, H_unroll , W_unroll, out_channel, W_unroll, yoffset, stream[i%stream_size], i%stream_size);

  }  

  cudaDeviceSynchronize();
  const auto end = now();


  for( const auto i : range(0, stream_size) )
    cudaStreamDestroy( stream[i] ); 

  const auto elapsed =
      std::chrono::duration<double, std::milli>(end - start).count();
  std::cout <<"Convolution needs: "<<elapsed<<" ms"<<std::endl;


}


static void w_transform(const float *W, const int wdims[4], float *W_t){
  const auto w_start = now();

  for(const auto p : range(0,wdims[0])){
    for(const auto q :range(0,wdims[1])){
      for(const auto c :range(0,wdims[2])){
        for(const auto m :range(0,wdims[3])){
          unsigned int woff = p* wdims[1]*wdims[2]*wdims[3]+ 
                              q* wdims[2]*wdims[3]+
                              c *wdims[3]+ m;
 
          unsigned int woff_t = c* wdims[0]*wdims[1]*wdims[3]+
                                p* wdims[1]*wdims[3]+
                                q* wdims[3]+ 
                               
                                 m;
          
          W_t[woff_t]=W[woff];
        }
      }
    }
  }
  const auto w_end = now();

  const auto w_elapsed =
      std::chrono::duration<double, std::milli>(w_end - w_start).count();
  std::cout <<"W_transfrom needs: "<<w_elapsed<<" ms"<<std::endl;
}





// Recified linear unit 2d
static void relu2(float *dX, const int xdims[2]) {
  int length = xdims[0] * xdims[1];
  const auto start  = now();


  int TILE_SIZE = 32;
  dim3 DimBlock( TILE_SIZE , 1 , 1 );
  dim3 DimGrid( length/TILE_SIZE + 1 , 1 , 1 );

  relu2_Kernel<<<DimGrid, DimBlock>>>( dX , length );

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) 
    printf("Error: %s\n", cudaGetErrorString(err));

  const auto end = now(); 
  const auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();
  std::cout <<"Relu2 needs: "<<elapsed<<" ms"<<std::endl;
}





static void call_average_pool_kernel(float *device_X, const int xdims[4],
                         const int pool_size, float *device_Y, const int ydims[4], const int block_size = 16){
  assert(ydims[1] == ydims[2]); //row == col
  int batch_size = ydims[0];
  int n_rows_o = ydims[1];
  int n_cols_o = ydims[2];
  int n_maps_o = ydims[3];
  int *device_xdims;
  int *device_ydims;
  cudaMalloc((void **) &device_xdims, sizeof(int) * 4);
  cudaMalloc((void **) &device_ydims, sizeof(int) * 4);
  cudaMemcpy(device_xdims, xdims, sizeof(int) * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(device_ydims, ydims, sizeof(int) * 4, cudaMemcpyHostToDevice);


  dim3 dimGrid(batch_size, n_maps_o, 1);
  dim3 dimBlock(n_rows_o, n_cols_o, 1);
  const auto tic1 = now();
  average_pool_kernel<<<dimGrid, dimBlock>>>(device_X, device_xdims, pool_size, device_Y, device_ydims);
  const auto toc1 = now();
  const auto elapsed1 = std::chrono::duration<double, std::milli>(toc1 - tic1).count();;
  std::cout << "avg_pool_kernel took " << elapsed1 << " ms\n";
  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess)
    printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));

}







static void fully_forward(float *dX, const int xdims[2], float *dW,
                          const int wdims[2], float *dY, const int ydims[2]) {

  int TILE_SIZE = 16;
  dim3 DimBlock(TILE_SIZE,TILE_SIZE,1);
  dim3 DimGrid( xdims[0]/TILE_SIZE + 1 , wdims[1]/TILE_SIZE + 1 , 1 );

  FullyForward_Kernel<<<DimGrid, DimBlock>>>( dX , dW , dY , xdims[1] , xdims[0] , wdims[1] );
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) 
    printf("Error: %s\n", cudaGetErrorString(err));
}





// Choose the guess with largest score
static void argmax(const float *X, const int xdims[2], int *Y) {
  for (const auto i : range(0, xdims[0])) {
    auto max_idx = 0;
    auto max     = X[i * xdims[1]];
    for (const auto j : range(0, xdims[1])) {
      const auto elem = X[(i * xdims[1]) + j];
      if (elem > max) {
        max_idx = j;
        max     = elem;
      }
    }
    Y[i] = max_idx;
  }
}

// Forward operation for the CNN, a combination of conv layer + average pooling
// + relu
void forward_operation(float *x, float *conv1, float *conv2, float *fc1,
                       float *fc2, int *out) {
  
  float *d_X1;
  cudaMalloc((void **)&d_X1, xdims[0] * xdims[1] * xdims[2] * xdims[3] * sizeof(float));
  cudaMemcpy(d_X1, x,  xdims[0] * xdims[1] * xdims[2] * xdims[3] * sizeof(float), cudaMemcpyHostToDevice);
  

  // 1st Convolution Layer (reduce to matrix multiplication)
  float *d_X1_t, *d_W1_t;
  cudaMalloc((void **)&d_X1_t, xdims[0] * xdims[1] * xdims[2] * xdims[3] * sizeof(float));
  cudaMalloc((void **)&d_W1_t, conv1dims[0] * conv1dims[1] * conv1dims[2] * conv1dims[3] * sizeof(float));
  float W1_t[conv1dims[0] * conv1dims[1] * conv1dims[2] * conv1dims[3]]; 
  w_transform(conv1, conv1dims, W1_t);
  cudaMemcpy(d_W1_t, W1_t,  conv1dims[0] * conv1dims[1] * conv1dims[2] * conv1dims[3] * sizeof(float), cudaMemcpyHostToDevice);
  
  const int adims[] = {xdims[0], (xdims[1] - conv1dims[0] + 1),
                       (xdims[2] - conv1dims[1] + 1), conv1dims[3]};
  cudaMemcpyToSymbol(xdims_cd, xdims, 4 * sizeof(int));
  cudaMemcpyToSymbol(wdims_cd, conv1dims, 4 * sizeof(int));
  cudaMemcpyToSymbol(ydims_cd, adims, 4 * sizeof(int));
  x_transform(xdims, d_X1, d_X1_t);

  float * a;
  cudaMalloc((void **)&a, adims[0]*adims[1]*adims[2]*adims[3] * sizeof(float));
  conv_mat(d_X1_t, xdims, d_W1_t, conv1dims, a, adims);

  cudaFree(d_X1_t);
  cudaFree(d_W1_t);

  // average pooling
  const int pool_size = 2;
  const int bdims[]   = {adims[0], adims[1] / pool_size, adims[2] / pool_size,
                       adims[3]};
  float *b; 
  cudaMalloc((void **)&b, bdims[0]*bdims[1]*bdims[2]*bdims[3] * sizeof(float));     
  call_average_pool_kernel(a, adims, pool_size, b, bdims);
  cudaFree(a);
  


  // 2nd Convolution Layer (reduce to matrix multiplication)
  float *d_X2_t, *d_W2_t;
  cudaMalloc((void **)&d_X2_t, bdims[0] * bdims[1] * bdims[2] * bdims[3] * sizeof(float));
  cudaMalloc((void **)&d_W2_t, conv2dims[0] * conv2dims[1] * conv2dims[2] * conv2dims[3] * sizeof(float));
  float W2_t[conv2dims[0] * conv2dims[1] * conv2dims[2] * conv2dims[3]]; 
  w_transform(conv2, conv2dims, W2_t);
  cudaMemcpy(d_W2_t, W2_t,  conv2dims[0] * conv2dims[1] * conv2dims[2] * conv2dims[3] * sizeof(float), cudaMemcpyHostToDevice);
  
  const int cdims[] = {bdims[0], (bdims[1] - conv2dims[0] + 1),
                       (bdims[2] - conv2dims[1] + 1), conv2dims[3]};
  cudaMemcpyToSymbol(xdims_cd, bdims, 4 * sizeof(int));
  cudaMemcpyToSymbol(wdims_cd, conv2dims, 4 * sizeof(int));
  cudaMemcpyToSymbol(ydims_cd, cdims, 4 * sizeof(int));
  x_transform(bdims, b, d_X2_t);

  float *c;
  cudaMalloc((void **)&c, cdims[0]*cdims[1]*cdims[2]*cdims[3] * sizeof(float));
  conv_mat(d_X2_t, bdims, d_W2_t, conv2dims, c, cdims);
  cudaFree(b);
  cudaFree(d_X2_t);
  cudaFree(d_W2_t);

  // average pooling
  const int ddims[] = {cdims[0], cdims[1] / pool_size, cdims[2] / pool_size,
                       cdims[3]};
  float *d; 
  cudaMalloc((void **)&d, ddims[0]*ddims[1]*ddims[2]*ddims[3] * sizeof(float));
  call_average_pool_kernel(c, cdims, pool_size, d, ddims);
  cudaFree(c);



  // reshape
  const int ddims2[] = {ddims[0], ddims[1] * ddims[2] * ddims[3]};

  const int edims[] = {ddims[0], fc1dims[1]};
  float * e;
  cudaMalloc((void **)&e, edims[0]*fc1dims[1]* sizeof(float));
  float * fc1d;
  cudaMalloc((void **)&fc1d, flattened_length(fc1dims)* sizeof(float));
  cudaMemcpy(fc1d, fc1, flattened_length(fc1dims)* sizeof(float), cudaMemcpyHostToDevice);

  fully_forward(d, ddims2, fc1d, fc1dims, e, edims);
  cudaFree(d);
  cudaFree(fc1d);

  // relu
  relu2(e, edims);
  
  const int fdims[] = {edims[0], fc2dims[1]};
  auto f_host       = zeros<float>(fdims);
  float *df, *fc2d;
  cudaMalloc((void **)&df, fdims[0]*fdims[1] * sizeof(float));
  cudaMemset((void*) df, 0, fdims[0]*fdims[1] * sizeof(float));

  cudaMalloc((void **)&fc2d, flattened_length(fc2dims) * sizeof(float));
  cudaMemcpy(fc2d, fc2, flattened_length(fc2dims)* sizeof(float), cudaMemcpyHostToDevice);

  fully_forward(e, edims, fc2d, fc2dims, df, fdims);
  cudaFree(e);
  cudaFree(fc2d);
  cudaMemcpy(f_host, df, fdims[0]*fdims[1] * sizeof(float),  cudaMemcpyDeviceToHost);
  cudaFree(df);

  argmax(f_host, fdims, out);

  delete[] f_host;
}

int main(int argc, char **argv) {

  if (argc != 3 && argc != 4) {
    std::cerr << "\n"
              << "This program performs the forward opertion step for "
                 "Convolutional Neural Network(CNN).  "
                 "Sample usage: \n"
              << argv[0]
              << " [../data/test10.hdf5] [../data/model.hdf5] [10]\n";
    return -1;
  }
  FLAGS_testdata = std::string(argv[1]);
  FLAGS_model    = std::string(argv[2]);
  if (argc == 3) {
    const std::map<std::string, int> default_batch_sizes{
        {"../data/test2.hdf5", 2},
        {"../data/test10.hdf5", 10},
        {"../data/test100.hdf5", 100},
        {"../data/testfull.hdf5", 10000}};
    const auto batch_size_in_map = default_batch_sizes.find(FLAGS_testdata);
    if (batch_size_in_map == default_batch_sizes.end()) {
      std::cerr << "\nERROR:: Unrecognized file " << FLAGS_testdata << " batch_size must be specified.\n";
      return -1;
    }
    FLAGS_batch_size = batch_size_in_map->second;
  } else if (argc == 4) {
    FLAGS_batch_size = atoi(argv[3]);
  }
  xdims[0] = FLAGS_batch_size;
  rdims[0] = FLAGS_batch_size;

  // Load data into x and y
  float *x = allocate<float>(xdims);
  float *y = allocate<float>(rdims);
  loadData(x, y);

  // Load model
  float *conv1 = allocate<float>(conv1dims);
  float *conv2 = allocate<float>(conv2dims);
  float *fc1   = allocate<float>(fc1dims);
  float *fc2   = allocate<float>(fc2dims);
  loadModel(conv1, conv2, fc1, fc2);

  // Perform foward opertion
  int *out = zeros<int>(FLAGS_batch_size);

  // get start time
  const auto start = now();

  forward_operation(x, conv1, conv2, fc1, fc2, out);

  // get end time
  const auto end = now();

  // get elapsed time in milliseconds
  const auto elapsed =
      std::chrono::duration<double, std::milli>(end - start).count();

  // Get reference
  int *ref = zeros<int>(FLAGS_batch_size);
  argmax(y, rdims, ref);

  // Calculate correctness
  int num_correct = 0;
  for (const auto i : range(0, FLAGS_batch_size)) {
    if (out[i] == ref[i]) {
      num_correct++;
    }
  }
  std::cout << "Done with " << FLAGS_batch_size << " queries in "
            << "elapsed = " << elapsed << " milliseconds. Correctness: "
            << static_cast<float>(num_correct) / FLAGS_batch_size << "\n";

  delete[] x;
  delete[] y;
  delete[] conv1;
  delete[] conv2;
  delete[] fc1;
  delete[] fc2;
  delete[] out;
  delete[] ref;

  return 0;
}
