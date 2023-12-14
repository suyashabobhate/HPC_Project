__global__ void ab_gpu(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int row = blockDim.x*blockIdx.x+threadIdx.x;
  int col = blockDim.y*blockIdx.y+threadIdx.y;
  float sum = 0;

  if((row < Ni) && (col < Nj)) {
    for (int k = 0; k < Nk; k++) {
      sum += A[row * Nk + k] * B[k * Nj + col];
    }
  
    C[row * Nj + col] = sum; 
  }
}


// k unroll //
__global__ void ab_gpu_kunroll(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int row = blockDim.x*blockIdx.x+threadIdx.x;
  int col = blockDim.y*blockIdx.y+threadIdx.y;
  float sum = 0;

if((row < Ni) && (col < Nj)) {
  for (int k = 0; k < Nk; k+=4) {
    sum += A[row * Nk + k] * B[k * Nj + col];
    sum += A[row * Nk + (k+1)] * B[(k+1) * Nj + col];
    sum += A[row * Nk + (k+2)] * B[(k+2) * Nj + col];
    sum += A[row * Nk + (k+3)] * B[(k+3) * Nj + col];
  }
  
  C[row * Nj + col] = sum; 
}
}

// j unroll by 4 //
__global__ void ab_gpu_junroll(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int row = blockDim.x*blockIdx.x+threadIdx.x;
  int col = 4*(blockDim.y*blockIdx.y+threadIdx.y);
  float sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;

  for (int k = 0; k < Nk; k++) {
    sum1 += A[row * Nk + k] * B[k * Nj + col];
    sum2 += A[row * Nk + k] * B[k * Nj + (col+1)];
    sum3 += A[row * Nk + k] * B[k * Nj + (col+2)];
    sum4 += A[row * Nk + k] * B[k * Nj + (col+3)];
  }
  
  C[row * Nj + col] = sum1;
  C[row * Nj + (col+1)] = sum2;
  C[row * Nj + (col+2)] = sum3;
  C[row * Nj + (col+3)] = sum4;

}

// i unroll //
__global__ void ab_gpu_iunroll(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int row = blockDim.x*blockIdx.x*4+threadIdx.x;
  int col = blockDim.y*blockIdx.y+threadIdx.y;
  float sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;

  for (int k = 0; k < Nk; k++) {
    sum1 += A[row * Nk + k] * B[k * Nj + col];
    sum2 += A[(row + blockDim.x) * Nk + k] * B[k * Nj + col];
    sum3 += A[(row + 2*blockDim.x)* Nk + k] * B[k * Nj + col];
    sum4 += A[(row + 3*blockDim.x) * Nk + k] * B[k * Nj + col];
  }
  
  C[row * Nj + col] = sum1;
  C[(row + blockDim.x) * Nj + col] = sum2;
  C[(row + 2*blockDim.x) * Nj + col] = sum3;
  C[(row + 3*blockDim.x) * Nj + col] = sum4;

}

// i and j unroll by 4 //
__global__ void ab_gpu_ijunroll(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int row = blockDim.x*blockIdx.x*4+threadIdx.x;
  int col = 4*(blockDim.y*blockIdx.y+threadIdx.y);

  float sum1 = 0, sum2 = 0, sum3 = 0, 
  sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0, 
  sum8 = 0, sum9 = 0, sum10 = 0, sum11 = 0, 
  sum12 = 0, sum13 = 0, sum14 = 0, sum15 = 0, sum16 = 0;

  for (int k = 0; k < Nk; k++) {
    sum1 += A[row * Nk + k] * B[k * Nj + col];
    sum2 += A[row * Nk + k] * B[k * Nj + (col+1)];
    sum3 += A[row * Nk + k] * B[k * Nj + (col+2)];
    sum4 += A[row * Nk + k] * B[k * Nj + (col+3)];

    sum5 += A[(row + blockDim.x) * Nk + k] * B[k * Nj + col];
    sum6 += A[(row + blockDim.x) * Nk + k] * B[k * Nj + (col+1)];
    sum7 += A[(row + blockDim.x) * Nk + k] * B[k * Nj + (col+2)];
    sum8 += A[(row + blockDim.x) * Nk + k] * B[k * Nj + (col+3)];

    sum9 += A[(row + 2*blockDim.x) * Nk + k] * B[k * Nj + col];
    sum10 += A[(row + 2*blockDim.x) * Nk + k] * B[k * Nj + (col+1)];
    sum11 += A[(row + 2*blockDim.x) * Nk + k] * B[k * Nj + (col+2)];
    sum12 += A[(row + 2*blockDim.x) * Nk + k] * B[k * Nj + (col+3)];

    sum13 += A[(row + 3*blockDim.x)* Nk + k] * B[k * Nj + col];
    sum14 += A[(row + 3*blockDim.x) * Nk + k] * B[k * Nj + (col+1)];
    sum15 += A[(row + 3*blockDim.x) * Nk + k] * B[k * Nj + (col+2)];
    sum16 += A[(row + 3*blockDim.x) * Nk + k] * B[k * Nj + (col+3)];
  }
  
  C[row * Nj + col] = sum1;
  C[row * Nj + (col+1)] = sum2;
  C[row * Nj + (col+2)] = sum3;
  C[row * Nj + (col+3)] = sum4;

  C[(row + blockDim.x) * Nj + col] = sum5;
  C[(row + blockDim.x) * Nj + (col+1)] = sum6;
  C[(row + blockDim.x) * Nj + (col+2)] = sum7;
  C[(row + blockDim.x) * Nj + (col+3)] = sum8;

  C[(row + 2*blockDim.x) * Nj + col] = sum9;
  C[(row + 2*blockDim.x) * Nj + (col+1)] = sum10;
  C[(row + 2*blockDim.x) * Nj + (col+2)] = sum11;
  C[(row + 2*blockDim.x) * Nj + (col+3)] = sum12;

  C[(row + 3*blockDim.x) * Nj + col] = sum13;
  C[(row + 3*blockDim.x) * Nj + (col+1)] = sum14;
  C[(row + 3*blockDim.x) * Nj + (col+2)] = sum15;
  C[(row + 3*blockDim.x) * Nj + (col+3)] = sum16;


}


// j unroll by 8 //
__global__ void ab_gpu_junroll8(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int row = blockDim.x*blockIdx.x+threadIdx.x;
  int col = 8*(blockDim.y*blockIdx.y+threadIdx.y);
  float sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0, sum8 = 0;

  for (int k = 0; k < Nk; k++) {
    sum1 += A[row * Nk + k] * B[k * Nj + col];
    sum2 += A[row * Nk + k] * B[k * Nj + (col+1)];
    sum3 += A[row * Nk + k] * B[k * Nj + (col+2)];
    sum4 += A[row * Nk + k] * B[k * Nj + (col+3)];
    sum5 += A[row * Nk + k] * B[k * Nj + (col+4)];
    sum6 += A[row * Nk + k] * B[k * Nj + (col+5)];
    sum7 += A[row * Nk + k] * B[k * Nj + (col+6)];
    sum8 += A[row * Nk + k] * B[k * Nj + (col+7)];
  }
  
  C[row * Nj + col] = sum1;
  C[row * Nj + (col+1)] = sum2;
  C[row * Nj + (col+2)] = sum3;
  C[row * Nj + (col+3)] = sum4;
  C[row * Nj + (col+4)] = sum5;
  C[row * Nj + (col+5)] = sum6;
  C[row * Nj + (col+6)] = sum7;
  C[row * Nj + (col+7)] = sum8;


}





// __global__ void ab_gpu(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
// {
//   int i, j, k;

//   for (i = 0; i < Ni; i++) 
//    for (j = 0; j < Nj; j++)
//     C[i*Nj+j]=0.0;
//   for (i = 0; i < Ni; i++)
//    for (j = 0; j < Nj; j++)
//     for (k = 0; k < Nk; k++)
// // C[i][j] = C[i][j] + A[i][k]*B[k][j];
//      C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[k*Nj+j];
// }