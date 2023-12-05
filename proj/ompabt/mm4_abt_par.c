#define min(X, Y) (((X) < (Y)) ? (X) : (Y))

void abt_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;

  #pragma omp parallel private(i,j,k) 
  {
    #pragma omp for schedule (static)
     for (i = 0; i < Ni; i++)
          for (j = 0; j < Nj; j++)
           for (k = 0; k < Nk; k++)
               // C[i][j] = C[i][j] + A[i][k]*B[j][k];
               C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[j*Nk+k];


  }
}

/////////// remaining to change logic from ab to abt
///////////////// k unrolling ///////////////////////////////////////////////
void abt_kunroll_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) {
     int i, j, k;

  #pragma omp parallel private(i,j,k) 
  {

     #pragma omp for schedule (static)
     for (i = 0; i < Ni; i++) {
       for (j = 0; j < Nj; j++) {
          int rem = Nj % 4;
          for (k = 0; k < Nk - rem; k+=4){
               C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[k*Nj+j];
               C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+(k+1)]*B[(k+1)*Nj+j];
               C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+(k+2)]*B[(k+2)*Nj+j];
               C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+(k+3)]*B[(k+3)*Nj+j];
          }
          for (k = Nk - rem; k < Nk; k++){
               C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[k*Nj+j];
          }
       }
     }
  }
}

///////////////// j unrolling ///////////////////////////////////////////////
void abt_junroll_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) {
     int i, j, k;

  #pragma omp parallel private(i,j,k) 
  {
     #pragma omp for schedule(static)
     for (i = 0; i < Ni; i++) {
       int rem  = Ni % 4;
       for (j = 0; j < Nj - rem; j+=4) {
          for (k = 0; k < Nk; k++){
               C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[k*Nj+j];
               C[i*Nj+(j+1)]=C[i*Nj+(j+1)]+A[i*Nk+k]*B[k*Nj+(j+1)];
               C[i*Nj+(j+2)]=C[i*Nj+(j+2)]+A[i*Nk+k]*B[k*Nj+(j+2)];
               C[i*Nj+(j+3)]=C[i*Nj+(j+3)]+A[i*Nk+k]*B[k*Nj+(j+3)];
          }
          // C[i*Nj+j]=C[i*Nj+j]+A[i*Nj+j]*B[j*Nj+j];
       }
       for (j = Nj - rem; j < Nj; j++) {       
          for (k = 0; k < Nk; k++){
               C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[k*Nj+j];
          }
       }
     }
  }
}

///////////////// all tiling ///////////////////////////////////////////////
void abt_alltile16_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) {
     int i, j, k, it, jt, kt;

  #pragma omp parallel private(i,j,k,it,jt,kt) 
  {

     #pragma omp for schedule (static)
     for (it = 0; it < Ni; it+=16) {
          for (jt = 0; jt < Nj; jt+=16) {
               for (kt = 0; kt < Nk; kt+=16){
                    for (i = it; i < min(it+16,Ni); i++) {
                         for (j = jt; j < min(jt+16,Nj); j++) {
                              for (k = kt; k < min(kt+16,Nk); k++){
                                   C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[k*Nj+j];
                              }
                         }
                    }
               }
          }
     }
  }

}