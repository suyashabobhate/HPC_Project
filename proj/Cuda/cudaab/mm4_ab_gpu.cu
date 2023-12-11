__global__ void ab_gpu(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int row = blockDim.x*blockIdx.x+threadIdx.x;
  int col = blockDim.y*blockIdx.y+threadIdx.y;
  float sum = 0;

  for (int k = 0; k < Nk; k++) {
    sum += A[row * Nk + k] * B[k * Nj + col];
  }
  
  C[row * Nj + col] = sum; 
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