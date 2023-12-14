#define min(X, Y) (((X) < (Y)) ? (X) : (Y))

void atb_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;

  #pragma omp parallel private(i,j,k) 
  {
    #pragma omp for schedule (static)
     for (i = 0; i < Ni; i++)
          for (j = 0; j < Nj; j++)
               for (k = 0; k < Nk; k++)
               // C[i][j] = C[i][j] + A[k][i]*B[k][j];
                    C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];


  }
}

///////////////// k unrolling ///////////////////////////////////////////////
void atb_kunroll_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) {
     int i, j, k;

  #pragma omp parallel private(i,j,k) 
  {

     #pragma omp for schedule (static)
     for (i = 0; i < Ni; i++) {
       for (j = 0; j < Nj; j++) {
          int rem = Nk % 4;
          for (k = 0; k < rem; k++){
               C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];
          }
          for (k = rem; k < Nk; k+=4){
               C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];
               C[i*Nj+j]=C[i*Nj+j]+A[(k+1)*Ni+i]*B[(k+1)*Nj+j];
               C[i*Nj+j]=C[i*Nj+j]+A[(k+2)*Ni+i]*B[(k+2)*Nj+j];
               C[i*Nj+j]=C[i*Nj+j]+A[(k+3)*Ni+i]*B[(k+3)*Nj+j];
          }
       }
     }
  }
}

///////////////// j unrolling ///////////////////////////////////////////////
void atb_junroll_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) {
     int i, j, k;

  #pragma omp parallel private(i,j,k) 
  {
     #pragma omp for schedule(static)
     for (i = 0; i < Ni; i++) {
       int rem  = Nj % 4;
       for (j = 0; j < rem; j++) {
          for (k = 0; k < Nk; k++){
               C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];
          }
       }
       for (j = rem; j < Nj; j+=4) {       
          for (k = 0; k < Nk; k++){
               C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];
               C[i*Nj+(j+1)]=C[i*Nj+(j+1)]+A[k*Ni+i]*B[k*Nj+(j+1)];
               C[i*Nj+(j+2)]=C[i*Nj+(j+2)]+A[k*Ni+i]*B[k*Nj+(j+2)];
               C[i*Nj+(j+3)]=C[i*Nj+(j+3)]+A[k*Ni+i]*B[k*Nj+(j+3)];
          }
       }
     }
  }
}

///////////////// all tiling ///////////////////////////////////////////////
void atb_alltile16_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) {
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
                                   C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];
                              }
                         }
                    }
               }
          }
     }
  }

}


// permute from ijk to ikj ////////////////////////////// 
void atb_permute_ikj_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) {
     int i, j, k;

  #pragma omp parallel private(i,j,k) 
  {
    #pragma omp for schedule (static)
     for (i = 0; i < Ni; i++)
          for (k = 0; k < Nk; k++)
               for (j = 0; j < Nj; j++)
                   C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];

  }

}

// permute from ijk to ijk ////////////////////////////// 
void atb_permute_ijk_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) {
     int i, j, k;

  #pragma omp parallel private(i,j,k) 
  {
    #pragma omp for schedule (static)
     for (i = 0; i < Ni; i++)
          for (j = 0; j < Nj; j++)
               for (k = 0; k < Nk; k++)
                    C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];

  }

}

// j unroll by 2 ////////////////////////////// 
void atb_junrollby2(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) {
     int i, j, k;

  #pragma omp parallel private(i,j,k) 
  {
    #pragma omp for schedule (static)
     for (i = 0; i < Ni; i++) {
          int rem = Nj % 2;
          for (j = 0; j < rem; j++) {
               for (k = 0; k < Nk; k++){
               C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];
          }
       }
       for (j = rem; j < Nj; j+=2) {       
          for (k = 0; k < Nk; k++){
               C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];
               C[i*Nj+(j+1)]=C[i*Nj+(j+1)]+A[k*Ni+i]*B[k*Nj+(j+1)];
               // C[i*Nj+(j+2)]=C[i*Nj+(j+2)]+A[i*Nk+k]*B[(j+2)*Nk+k];
               // C[i*Nj+(j+3)]=C[i*Nj+(j+3)]+A[i*Nk+k]*B[(j+3)*Nk+k];
               // C[i*Nj+(j+4)]=C[i*Nj+(j+4)]+A[i*Nk+k]*B[(j+4)*Nk+k];
               // C[i*Nj+(j+5)]=C[i*Nj+(j+5)]+A[i*Nk+k]*B[(j+5)*Nk+k];
               // C[i*Nj+(j+6)]=C[i*Nj+(j+6)]+A[i*Nk+k]*B[(j+6)*Nk+k];
               // C[i*Nj+(j+7)]=C[i*Nj+(j+7)]+A[i*Nk+k]*B[(j+7)*Nk+k];
          }
          }
     }

  }

}

// permute from ijk to kij ////////////////////////////// 
void atb_permute_kij_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) {
     int i, j, k;

  #pragma omp parallel private(i,j,k) 
  {
     for (k = 0; k < Nk; k++)
     #pragma omp for schedule (static)
          for (i = 0; i < Ni; i++)
               for (j = 0; j < Nj; j++)
                    C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];

  }

}

// add pragma omp for on j ////////////////////////////// 
void atb_paralellonj_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) {
     int i, j, k;

  #pragma omp parallel private(i,j,k) 
  {  
     for (i = 0; i < Ni; i++)
          #pragma omp for schedule (static)
          for (j = 0; j < Nj; j++)
               for (k = 0; k < Nk; k++)
                    C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];

  }

}

// add pragma omp for on j using ikj permutation ////////////////////////////// 
void atb_paralellonj_permute_ikj_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) {
     int i, j, k;

  #pragma omp parallel private(i,j,k) 
  {  
     for (i = 0; i < Ni; i++)
          for (k = 0; k < Nk; k++)
               #pragma omp for schedule (static)
               for (j = 0; j < Nj; j++)
                    C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];

  }

}


// j unroll on ikj permutation ////////////////////////////// 
void atb_junroll_permute_ikj_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) {
     int i, j, k;

  #pragma omp parallel private(i,j,k) 
  {  
     #pragma omp for schedule (static)
     for (i = 0; i < Ni; i++) {
          for (k = 0; k < Nk; k++) {
               int rem = Nj % 4;
               for (j = 0; j < rem; j++) {
                    C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];
               }
               for (j = rem; j < Nj; j+=4) {
                    C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];
                    C[i*Nj+(j+1)]=C[i*Nj+(j+1)]+A[k*Ni+i]*B[k*Nj+(j+1)];
                    C[i*Nj+(j+2)]=C[i*Nj+(j+2)]+A[k*Ni+i]*B[k*Nj+(j+2)];
                    C[i*Nj+(j+3)]=C[i*Nj+(j+3)]+A[k*Ni+i]*B[k*Nj+(j+3)];
               }
          }  
          
     }
  }

}

// k unroll on ikj permutation ////////////////////////////// 
void atb_kunroll_permute_ikj_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) {
     int i, j, k;

  #pragma omp parallel private(i,j,k) 
  {  
     #pragma omp for schedule (static)
     for (i = 0; i < Ni; i++) {
          int rem = Nk % 4;
          for (k = 0; k < rem; k++) {
               for (j = 0; j < Nj; j++) {
                    C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];
               }
          } 

          for (k = rem; k < Nk; k+=4) {
               for (j = 0; j < Nj; j++) {
                    C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];
                    C[i*Nj+j]=C[i*Nj+j]+A[(k+1)*Ni+i]*B[(k+1)*Nj+j];
                    C[i*Nj+j]=C[i*Nj+j]+A[(k+2)*Ni+i]*B[(k+2)*Nj+j];
                    C[i*Nj+j]=C[i*Nj+j]+A[(k+3)*Ni+i]*B[(k+3)*Nj+j];
               } 
          }
          
     }
  }

}

// k unroll on kij permutation ////////////////////////////// 
void atb_kunroll_permute_kij_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) {
     int i, j, k;

  #pragma omp parallel private(i,j,k) 
  {  
     for (k = 0; k < Nk; k+=4) {
          #pragma omp for schedule (static)
          for (i = 0; i < Ni; i++) {
               for (j = 0; j < Nj; j++) {
                    C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[k*Nj+j];
                    C[i*Nj+j]=C[i*Nj+j]+A[(k+1)*Ni+i]*B[(k+1)*Nj+j];
                    C[i*Nj+j]=C[i*Nj+j]+A[(k+2)*Ni+i]*B[(k+2)*Nj+j];
                    C[i*Nj+j]=C[i*Nj+j]+A[(k+3)*Ni+i]*B[(k+3)*Nj+j];
               }
          } 
                    

     }
          
  }

}