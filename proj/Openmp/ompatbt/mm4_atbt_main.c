// Use "gcc -O3 -fopenmp mm4_main.c mm4_par.c " to compile 

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#define NTrials (5)
#define threshold (0.0000001)

void atbt_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk);
void atbt_junroll_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk);
// void atbt_kunroll_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk);
void atbt_alltile16_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk);
void atbt_permute_ikj_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk);
void atbt_permute_kij_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk);
// void atbt_paralellonj_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk);
// void atbt_paralellonj_permute_ikj_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk);
void atbt_junroll_permute_ikj_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk);
void atbt_kunroll_permute_ikj_par(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk);
void atbt_junrollby2(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk);

void atbt_seq(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int Ni, int Nj, int Nk)
{
  int i, j, k;

  for (i = 0; i < Ni; i++)
   for (j = 0; j < Nj; j++)
    for (k = 0; k < Nk; k++)
// C[i][j] = C[i][j] + A[i][k]*B[k][j];
     C[i*Nj+j]=C[i*Nj+j]+A[k*Ni+i]*B[j*Nk+k];
}

int main(int argc, char *argv[]){
  double tstart,telapsed;

  double mint_par[3],maxt_par[3];
  double mint_seq,maxt_seq;

  float *A, *B, *C, *Cref;
  int i,j,k,nt,trial,max_threads,num_cases;
  int nthreads[1];
  
  int Ni = atoi(argv[1]);
  int Nj = atoi(argv[2]);
  int Nk = atoi(argv[3]);

  A = (float *) malloc(sizeof(float)*Ni*Nk);
  B = (float *) malloc(sizeof(float)*Nk*Nj);
  C = (float *) malloc(sizeof(float)*Ni*Nj);
  Cref = (float *) malloc(sizeof(float)*Ni*Nj);
  for (i=0; i<Ni; i++)
   for (k=0; k<Nk; k++)
    A[k*Ni+i] = rand();
  for (k=0; k<Nk; k++)
   for (j=0; j<Nj; j++)
    B[k*Nj+j] = rand();

  max_threads = omp_get_max_threads();
  printf("Max Threads (from omp_get_max_threads) = %d\n",max_threads);
  num_cases = 1; nthreads[0] = max_threads - 1;

  for(int version=0; version<8;version++)
  {
    printf("Reference sequential code performance for ATBT (in GFLOPS)");
    mint_seq = 1e9; maxt_seq = 0;
    for(trial=0;trial<NTrials;trial++)
    {
     for(i=0;i<Ni;i++) for(j=0;j<Nj;j++) {C[i*Nj+j] = 0; Cref[i*Nj+j] = 0;}
     tstart = omp_get_wtime();
     atbt_seq(A,B,Cref,Ni,Nj,Nk);
     telapsed = omp_get_wtime()-tstart;
     if (telapsed < mint_seq) mint_seq=telapsed;
     if (telapsed > maxt_seq) maxt_seq=telapsed;
    }
    printf(" Min: %.2f; Max: %.2f\n",2.0e-9*Ni*Nj*Nk/maxt_seq,2.0e-9*Ni*Nj*Nk/mint_seq);

    for (nt=0;nt<num_cases;nt ++)
    {
     omp_set_num_threads(nthreads[nt]);
     mint_par[nt] = 1e9; maxt_par[nt] = 0;
     for (trial=0;trial<NTrials;trial++)
     {
      for(i=0;i<Ni;i++) for(j=0;j<Nj;j++) C[i*Nj+j] = 0;
      tstart = omp_get_wtime();
      switch (version) {
      case 0: atbt_par(A,B,C,Ni,Nj,Nk); break;
      case 1: atbt_junroll_par(A,B,C,Ni,Nj,Nk); break;
      // case 2: atbt_kunroll_par(A,B,C,Ni,Nj,Nk); break;
      case 2: atbt_alltile16_par(A,B,C,Ni,Nj,Nk); break;
      case 3: atbt_permute_ikj_par(A,B,C,Ni,Nj,Nk); break;
      case 4: atbt_permute_kij_par(A,B,C,Ni,Nj,Nk); break;
      // case 6: atbt_paralellonj_par(A,B,C,Ni,Nj,Nk); break;
      // case 7: atbt_paralellonj_permute_ikj_par(A,B,C,Ni,Nj,Nk); break;
      case 5: atbt_junroll_permute_ikj_par(A,B,C,Ni,Nj,Nk); break;
      case 6: atbt_kunroll_permute_ikj_par(A,B,C,Ni,Nj,Nk); break;
      case 7: atbt_junrollby2(A,B,C,Ni,Nj,Nk); break;
      }
    telapsed = omp_get_wtime()-tstart;
    if (telapsed < mint_par[nt]) mint_par[nt]=telapsed;
    if (telapsed > maxt_par[nt]) maxt_par[nt]=telapsed;
for (int l = 0; l < Ni*Nj; l++) if (fabs((C[l] - Cref[l])/Cref[l])>threshold) {printf("Error: mismatch at linearized index %d, was: %f, should be: %f for version %d\n ", l, C[l], Cref[l], version); return -1;}
   }
  }
    switch (version) {
    case 0: printf("Performance (Best & Worst) of parallel version for atbt_par (in GFLOPS)"); break;
    case 1: printf("Performance (Best & Worst) of parallel version for atbt_junroll_par (in GFLOPS)"); break;
    // case 2: printf("Performance (Best & Worst) of parallel version for atbt_kunroll_par (in GFLOPS)"); break;
    case 2: printf("Performance (Best & Worst) of parallel version for atbt_alltile16_par (in GFLOPS)"); break;
    case 3: printf("Performance (Best & Worst) of parallel version for atbt_permute_ikj_par (in GFLOPS)"); break;
    case 4: printf("Performance (Best & Worst) of parallel version for atbt_permute_kij_par (in GFLOPS)"); break;
    // case 6: printf("Performance (Best & Worst) of parallel version for atbt_paralellonj_par (in GFLOPS)"); break;
    // case 7: printf("Performance (Best & Worst) of parallel version for atbt_paralellonj_permute_ikj_par (in GFLOPS)"); break;
    case 5: printf("Performance (Best & Worst) of parallel version for atbt_junroll_permute_ikj_par (in GFLOPS)"); break;
    case 6: printf("Performance (Best & Worst) of parallel version for atbt_kunroll_permute_ikj_par (in GFLOPS)"); break;
    case 7: printf("Performance (Best & Worst) of parallel version for atbt_junrollby2 (in GFLOPS)"); break;

    }
    for (nt=0;nt<num_cases-1;nt++) printf("%d/",nthreads[nt]);
    printf(" using %d threads\n",nthreads[num_cases-1]);
    printf("Best Performance (GFLOPS): ");
    for (nt=0;nt<num_cases;nt++) printf("%.2f ",2.0e-9*Ni*Nj*Nk/mint_par[nt]);
    printf("\n");
    printf("Worst Performance (GFLOPS): ");
    for (nt=0;nt<num_cases;nt++) printf("%.2f ",2.0e-9*Ni*Nj*Nk/maxt_par[nt]);
    printf("\n");
  }
}
