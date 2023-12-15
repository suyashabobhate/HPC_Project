#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#define threshold 0.0001
#define BLOCK_SIZE 16
#define FIXME 1

void checkCUDAError(const char *msg);

cudaEvent_t start, stop;
float tstart, elapsedTime;

__global__ void atbt_gpu(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk);
__global__ void atbt_gpu_kunroll(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk);
__global__ void atbt_gpu_junroll(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk);
__global__ void atbt_gpu_iunroll(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk);
__global__ void atbt_gpu_junroll8(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk);
__global__ void atbt_gpu_ijunroll(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk);

void aTbT_seq(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;

  for (i = 0; i < Ni; i++)
   for (j = 0; j < Nj; j++)
    for (k = 0; k < Nk; k++)
// C[i][j] = C[i][j] + A[k][i]*B[j][k];
     C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[j*Nk+k];
}

int main(int argc, char *argv[]){

  float *h_A, *h_B, *h_C, *h_Cref, *d_A, *d_B, *d_C;
  int i,j,k;
  int Ni, Nj, Nk;

  if(argc >= 3) {
        Ni = atoi(argv[1]);
        Nj = atoi(argv[2]);
        Nk = atoi(argv[3]);
  }

  h_A = (float *) malloc(sizeof(float)*Ni*Nk);
  h_B = (float *) malloc(sizeof(float)*Nk*Nj);
  h_C = (float *) malloc(sizeof(float)*Ni*Nj);
  h_Cref = (float *) malloc(sizeof(float)*Ni*Nj);;

  for (i=0; i<Ni; i++)
   for (k=0; k<Nk; k++)
    h_A[k*Ni+i] = rand();
  for (k=0; k<Nk; k++)
   for (j=0; j<Nj; j++)
    h_B[k*Nj+j] = rand();

  
 // Allocate device memory and copy input data over to GPU
  cudaMalloc(&d_A, Ni*Nk*sizeof(float));
  cudaMalloc(&d_B, Nk*Nj*sizeof(float));
  cudaMalloc(&d_C, Ni*Nj*sizeof(float));
  checkCUDAError("cudaMalloc failure");
  cudaMemcpy(d_A, h_A, Ni*Nk*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, Nk*Nj*sizeof(float), cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy H2D transfer failure");

  dim3 block(BLOCK_SIZE,BLOCK_SIZE);  
  dim3 grid(ceil(Ni/(float)BLOCK_SIZE),ceil(Nj/(float)BLOCK_SIZE));
  dim3 grid2(ceil(Ni/(float)BLOCK_SIZE),ceil(Nj/4/(float)BLOCK_SIZE)); // j unroll 4
  dim3 grid3(ceil(Ni/4/(float)BLOCK_SIZE),ceil(Nj/(float)BLOCK_SIZE)); // i unroll 4
  dim3 grid4(ceil(Ni/(float)BLOCK_SIZE),ceil(Nj/8/(float)BLOCK_SIZE)); // j unroll 8
  dim3 grid5(ceil(Ni/4/(float)BLOCK_SIZE),ceil(Nj/4/(float)BLOCK_SIZE)); // ij unroll 4
  for(int version=0; version<6; version++)
  {
   for(i=0;i<Ni;i++) for(j=0;j<Nj;j++) h_Cref[i*Nj+j] = 0;
   aTbT_seq(h_A,h_B,h_Cref,Ni,Nj,Nk);
    for(int trial=0;trial<1;trial++)
    {
     for(i=0;i<Ni;i++) for(j=0;j<Nj;j++) h_C[i*Nj+j] = 0; 
      printf("Trial %d: ",trial);
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start);
      // Launch kernel
      switch (version) {
      case 0: atbt_gpu<<<grid, block>>>(d_A, d_B, d_C,Ni,Nj,Nk); printf("ATBT "); break;
      case 1: atbt_gpu_kunroll<<<grid, block>>>(d_A, d_B, d_C,Ni,Nj,Nk); printf("ATBT K Unroll ");break;
      case 2: atbt_gpu_junroll<<<grid2, block>>>(d_A, d_B, d_C,Ni,Nj,Nk); printf("ATBT J Unroll ");break;
      case 3: atbt_gpu_iunroll<<<grid3, block>>>(d_A, d_B, d_C,Ni,Nj,Nk); printf("ATBT I Unroll "); break;
      case 4: atbt_gpu_junroll8<<<grid4, block>>>(d_A, d_B, d_C,Ni,Nj,Nk); printf("ATBT J Unroll by 8 "); break;
      case 5: atbt_gpu_ijunroll<<<grid5, block>>>(d_A, d_B, d_C,Ni,Nj,Nk); printf("ATBT IJ Unroll by 4 "); break;
      }
      checkCUDAError("GPU kernel launch failure");
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&elapsedTime, start,stop);
      cudaDeviceSynchronize();
      // Copy results back to host
      cudaMemcpy(h_C, d_C, Ni*Nj*sizeof(float), cudaMemcpyDeviceToHost);
      checkCUDAError("cudaMemcpy D2H");
      for (int i = 0; i < Ni*Nj; i++) if (fabs((h_C[i]-h_Cref[i])/h_Cref[i])>threshold) {printf("Error: mismatch at linearized index %d, was: %f, should be: %f\n", i, h_C[i], h_Cref[i]); return -1;}
      printf("GFLOPS: %.2f\n",2.0e-6*Ni*Nj*Nk/elapsedTime);
     }
  }
  return 0;
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

