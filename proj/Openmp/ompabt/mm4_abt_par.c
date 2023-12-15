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

///////////////// k unrolling ///////////////////////////////////////////////
// void abt_kunroll_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) {
//      int i, j, k;

//   #pragma omp parallel private(i,j,k) 
//   {

//      #pragma omp for schedule (static)
//      for (i = 0; i < Ni; i++) {
//        for (j = 0; j < Nj; j++) {
//           int rem = Nk % 4;
//           for (k = 0; k < rem; k++){
//                C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[j*Nk+k];
//           }
//           for (k = rem; k < Nk; k+=4){
//                C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[j*Nk+k];
//                C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+(k+1)]*B[j*Nk+(k+1)];
//                C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+(k+2)]*B[j*Nk+(k+2)];
//                C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+(k+3)]*B[j*Nk+(k+3)];
//           }
//        }
//      }
//   }
// }

///////////////// j unrolling ///////////////////////////////////////////////
void abt_junroll_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) {
     int i, j, k;

  #pragma omp parallel private(i,j,k) 
  {
     #pragma omp for schedule(static)
     for (i = 0; i < Ni; i++) {
       int rem  = Nj % 4;
       for (j = 0; j < rem; j++) {
          for (k = 0; k < Nk; k++){
               C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[j*Nk+k];
          }
          // C[i*Nj+j]=C[i*Nj+j]+A[i*Nj+j]*B[j*Nj+j];
       }
       for (j = rem; j < Nj; j+=4) {       
          for (k = 0; k < Nk; k++){
               C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[j*Nk+k];
               C[i*Nj+(j+1)]=C[i*Nj+(j+1)]+A[i*Nk+k]*B[(j+1)*Nk+k];
               C[i*Nj+(j+2)]=C[i*Nj+(j+2)]+A[i*Nk+k]*B[(j+2)*Nk+k];
               C[i*Nj+(j+3)]=C[i*Nj+(j+3)]+A[i*Nk+k]*B[(j+3)*Nk+k];
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
                                   C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[j*Nk+k];
                              }
                         }
                    }
               }
          }
     }
  }

}


// permute from ijk to ikj ////////////////////////////// 
void abt_permute_ikj_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) {
     int i, j, k;

  #pragma omp parallel private(i,j,k) 
  {
    #pragma omp for schedule (static)
     for (i = 0; i < Ni; i++)
          for (k = 0; k < Nk; k++)
               for (j = 0; j < Nj; j++)
                   C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[j*Nk+k];

  }

}

// permute from ijk to ijk ////////////////////////////// 
void abt_permute_ijk_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) {
     int i, j, k;

  #pragma omp parallel private(i,j,k) 
  {
    #pragma omp for schedule (static)
     for (i = 0; i < Ni; i++)
          for (j = 0; j < Nj; j++)
               for (k = 0; k < Nk; k++)
                    C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[j*Nk+k];

  }

}

// j unroll by 2 ////////////////////////////// 
void abt_junrollby2(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) {
     int i, j, k;

  #pragma omp parallel private(i,j,k) 
  {
    #pragma omp for schedule (static)
     for (i = 0; i < Ni; i++) {
          int rem = Nj % 2;
          for (j = 0; j < rem; j++) {
               for (k = 0; k < Nk; k++){
                    C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[j*Nk+k];
          }
       }
       for (j = rem; j < Nj; j+=2) {       
          for (k = 0; k < Nk; k++){
               C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[j*Nk+k];
               C[i*Nj+(j+1)]=C[i*Nj+(j+1)]+A[i*Nk+k]*B[(j+1)*Nk+k];
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
void abt_permute_kij_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) {
     int i, j, k;

  #pragma omp parallel private(i,j,k) 
  {
     for (k = 0; k < Nk; k++)
     #pragma omp for schedule (static)
          for (i = 0; i < Ni; i++)
               for (j = 0; j < Nj; j++)
                    C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[j*Nk+k];

  }

}

// add pragma omp for on j ////////////////////////////// 
// void abt_paralellonj_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) {
//      int i, j, k;

//   #pragma omp parallel private(i,j,k) 
//   {  
//      for (i = 0; i < Ni; i++)
//           #pragma omp for schedule (static)
//           for (j = 0; j < Nj; j++)
//                for (k = 0; k < Nk; k++)
//                     C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[j*Nk+k];

//   }

// }

// add pragma omp for on j using ikj permutation ////////////////////////////// 
// void abt_paralellonj_permute_ikj_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) {
//      int i, j, k;

//   #pragma omp parallel private(i,j,k) 
//   {  
//      for (i = 0; i < Ni; i++)
//           for (k = 0; k < Nk; k++)
//                #pragma omp for schedule (static)
//                for (j = 0; j < Nj; j++)
//                     C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[j*Nk+k];

//   }

// }


// j unroll on ikj permutation ////////////////////////////// 
void abt_junroll_permute_ikj_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) {
     int i, j, k;

  #pragma omp parallel private(i,j,k) 
  {  
     #pragma omp for schedule (static)
     for (i = 0; i < Ni; i++) {
          for (k = 0; k < Nk; k++) {
               int rem = Nj % 4;
               for (j = 0; j < rem; j++) {
                    C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[j*Nk+k];
               }
               for (j = rem; j < Nj; j+=4) {
                    C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[j*Nk+k];
                    C[i*Nj+(j+1)]=C[i*Nj+(j+1)]+A[i*Nk+k]*B[(j+1)*Nk+k];
                    C[i*Nj+(j+2)]=C[i*Nj+(j+2)]+A[i*Nk+k]*B[(j+2)*Nk+k];
                    C[i*Nj+(j+3)]=C[i*Nj+(j+3)]+A[i*Nk+k]*B[(j+3)*Nk+k];
               }
          }  
          
     }
  }

}

// k unroll on ikj permutation ////////////////////////////// 
void abt_kunroll_permute_ikj_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) {
     int i, j, k;

  #pragma omp parallel private(i,j,k) 
  {  
     #pragma omp for schedule (static)
     for (i = 0; i < Ni; i++) {
          int rem = Nk % 4;
          for (k = 0; k < rem; k++) {
               for (j = 0; j < Nj; j++) {
                    C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[j*Nk+k];
               }
          } 

          for (k = rem; k < Nk; k+=4) {
               for (j = 0; j < Nj; j++) {
                    C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[j*Nk+k];
                    C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+(k+1)]*B[j*Nk+(k+1)];
                    C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+(k+2)]*B[j*Nk+(k+2)];
                    C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+(k+3)]*B[j*Nk+(k+3)];
               } 
          }
          
     }
  }

}

// k and j unroll on ikj permutation ////////////////////////////// 
// void abt_kjunroll_permute_ikj_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk) {
//      int i, j, k;

//   #pragma omp parallel private(i,j,k) 
//   {  
//      #pragma omp for schedule (static)
//      for (i = 0; i < Ni; i++) {
//           for (k = 0; k < Nk; k+=4) {
//                for (j = 0; j < Nj; j+=4) {
//                     C[i*Nj+j]=C[i*Nj+j]+A[i*Nk+k]*B[k*Nj+j];
//                     C[i*Nj+(j+1)]=C[i*Nj+(j+1)]+A[i*Nk+k]*B[k*Nj+(j+1)];
//                     C[i*Nj+(j+2)]=C[i*Nj+(j+2)]+A[i*Nk+k]*B[k*Nj+(j+2)];
//                     C[i*Nj+(j+3)]=C[i*Nj+(j+3)]+A[i*Nk+k]*B[k*Nj+(j+3)];

//                     C[i*Nj+(j+4)]=C[i*Nj+(j+4)]+A[i*Nk+(k+1)]*B[(k+1)*Nj+j];
//                     C[i*Nj+(j+5)]=C[i*Nj+(j+5)]+A[i*Nk+(k+1)]*B[(k+1)*Nj+(j+1)];
//                     C[i*Nj+(j+6)]=C[i*Nj+(j+6)]+A[i*Nk+(k+1)]*B[(k+1)*Nj+(j+2)];
//                     C[i*Nj+(j+7)]=C[i*Nj+(j+7)]+A[i*Nk+(k+1)]*B[(k+1)*Nj+(j+3)];

//                     C[i*Nj+(j+8)]=C[i*Nj+(j+8)]+A[i*Nk+(k+2)]*B[(k+2)*Nj+j];
//                     C[i*Nj+(j+9)]=C[i*Nj+(j+9)]+A[i*Nk+(k+2)]*B[(k+2)*Nj+(j+1)];
//                     C[i*Nj+(j+10)]=C[i*Nj+(j+10)]+A[i*Nk+(k+2)]*B[(k+2)*Nj+(j+2)];
//                     C[i*Nj+(j+11)]=C[i*Nj+(j+11)]+A[i*Nk+(k+2)]*B[(k+2)*Nj+(j+3)];

//                     C[i*Nj+(j+12)]=C[i*Nj+(j+12)]+A[i*Nk+(k+3)]*B[(k+3)*Nj+j];
//                     C[i*Nj+(j+13)]=C[i*Nj+(j+13)]+A[i*Nk+(k+3)]*B[(k+3)*Nj+(j+1)];
//                     C[i*Nj+(j+14)]=C[i*Nj+(j+14)]+A[i*Nk+(k+3)]*B[(k+3)*Nj+(j+2)];
//                     C[i*Nj+(j+15)]=C[i*Nj+(j+15)]+A[i*Nk+(k+3)]*B[(k+3)*Nj+(j+3)];
//                }
               
              
//           } 
          
//      }
//   }

// }

