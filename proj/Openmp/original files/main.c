// Use "gcc -O3 -fopenmp mm4_main.c mm4_par.c " to compile 

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#define threshold (0.0000001)

int main(int argc, char *argv[]){
  double tstart,telapsed;

  double mint_par[3],maxt_par[3];
  double mint_seq,maxt_seq;

  int nt,trial,max_threads,num_cases;
  int nthreads[3];
  int Ni, Nj, Nk;
  char command[1000];

  printf("Specify Matrix dimension Ni, Nj, Nk: ");
  scanf("%d %d %d", &Ni,&Nj,&Nk);

  max_threads = omp_get_max_threads();
  num_cases = 3; nthreads[0]=1; nthreads[1] = max_threads/2 - 1; nthreads[2] = max_threads - 1;

  for(int version=0; version<4;version++)
  { 
    for (nt=0;nt<num_cases;nt ++)
    {
     omp_set_num_threads(nthreads[nt]);
     mint_par[nt] = 1e9; maxt_par[nt] = 0;
      tstart = omp_get_wtime();
      switch (version) {
      case 0: printf("For AB for trial: %d \n", nt+1); sprintf(command, "clang -O3 -fopenmp -o abtest ../ompab/mm4_ab_main.c ../ompab/mm4_ab_par.c; ./abtest %d %d %d", Ni, Nj, Nk); system(command); break;
      case 1: printf("For ABT for trial: %d \n", nt+1); sprintf(command, "clang -O3 -fopenmp -o abttest ../ompabt/mm4_abt_main.c ../ompabt/mm4_abt_par.c; ./abttest %d %d %d", Ni, Nj, Nk); system(command); break;
      case 2: printf("For ATB for trial: %d \n", nt+1); sprintf(command, "clang -O3 -fopenmp -o atbtest ../ompatb/mm4_atb_main.c ../ompatb/mm4_atb_par.c; ./atbtest %d %d %d", Ni, Nj, Nk); system(command); break;
      case 3: printf("For ATBT for trial: %d \n", nt+1); sprintf(command, "clang -O3 -fopenmp -o atbttest ../ompatbt/mm4_atbt_main.c ../ompatbt/mm4_atbt_par.c; ./atbttest %d %d %d", Ni, Nj, Nk); system(command); break;
      }
    telapsed = omp_get_wtime()-tstart;
    if (telapsed < mint_par[nt]) mint_par[nt]=telapsed;
    if (telapsed > maxt_par[nt]) maxt_par[nt]=telapsed;
    }
  
    switch (version) {
    case 0: printf("Performance (Best & Worst) of parallel version for AB (in GFLOPS)"); break;
    case 1: printf("Performance (Best & Worst) of parallel version for ATB (in GFLOPS)"); break;
    case 2: printf("Performance (Best & Worst) of parallel version for ABT (in GFLOPS)"); break;
    case 3: printf("Performance (Best & Worst) of parallel version for ATBT (in GFLOPS)"); break;
    }
    for (nt=0;nt<num_cases;nt++) printf("%d/",nthreads[nt]);
    printf(" using %d threads\n",nthreads[num_cases-1]);
    printf("Best Performance (GFLOPS): ");
    for (nt=0;nt<num_cases;nt++) printf("%.2f ",2.0e-9*Ni*Nj*Nk/mint_par[nt]);
    printf("\n");
    printf("Worst Performance (GFLOPS): ");
    for (nt=0;nt<num_cases;nt++) printf("%.2f ",2.0e-9*Ni*Nj*Nk/maxt_par[nt]);
    printf("\n");
  }
}
