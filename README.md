# HPC_Project
This repo contains my work for the HPC project on OpenMP and CUDA

## Code structure
The OpenMP files are stored in the *Openmp* folder, and the CUDA files are stored in the *Cuda* folder.

### OPENMP
All the variants, namely AB, ABT, ATB, and ATBT, have separate folders containing their own main files to calculate the GFLOPs and par files, which contain the OMP code with different functions. The results are stored in the files starting with *output*. There are a total of 8 output files for 8 test cases. These test cases are chosen to have an equal distribution of odd and even dimensions and with an increase/decrease of size with Ni, Nj, and Nk, which are the matrix dimensions. The *original files* folder contains the *main.c* file, which is the driving file for each variant's main files. It also contains all executables generated from the runtime. 

### CUDA
Each variant, AB, ABT, ATB, and ATBT, has a separate folder. Each folder contains the main file, which calculates the GFLOPs; the GPU file, which contains different kernels; the executables generated from the runtime; and the output file, which displays the output for all test cases of that variant.

## How to run the code
### OPENMP
1. Go to the location where the *main.c* file is present. This file is the driving file for the OMP main files for all variants of matrix multiplication.
2. Steps :
   
   - Enter the correct folder
   ```
   cd proj/Openmp/original\ files/
   ```
   - Compile the *main.c* file
   ```
   clang -O3 -fopenmp -o maintempFinal main.c
   ```
   - You can enter any number in place of 8192 8192 16, which represents the dimensions of the matrix, namely Ni, Nj, and NK.
   ```
   ./maintempFinal 8192 8192 16
   ```
   
### CUDA
1. The main files for running all variants are located in separate folders. To run, follow the steps below
2. Steps :

   - Enter the correct folder. You can enter any variant name instead of *cudaab* to run the main file of that variant instead.
   ```
   cd proj/Cuda/cudaab
   ```
   - Compile the main file. You can enter any variant name instead of *mm4_ab* to compile that variant's main file and the GPU file instead.
   ```
   nvcc -O3 mm4_ab_main.cu mm4_ab_gpu.cu
   ```
   - You can enter any number in place of 1024 1024 1024, which represents the dimensions of the matrix, namely Ni, Nj and NK.
   ```
   ./a.out 1024 1024 1024
   ```

## Results
### OPENMP
The results obtained on the CADE machine are stored in the *output files* inside the *Openmp folder*, i.e., follow the steps to locate the folder with all the CADE output files.

Steps :

1. Enter the correct folder.
```
cd proj/Openmp
```
2. Locate the below-mentioned output files.
```
ls
```

These are namely:
- **output** - Output file for all functions of all variants with input test case 1024 1024 1024
- **output2** - Output file for all functions of all variants with input test case 4096 4096 64
- **output3** - Output file for all functions of all variants with input test case 8192 8192 16
- **output4** - Output file for all functions of all variants with input test case 999 999 999
- **output5** - Output file for all functions of all variants with input test case 37 37 728271
- **output6** - Output file for all functions of all variants with input test case 333 333 8991
- **output7** - Output file for all functions of all variants with input test case 2997 2997 111
- **output8** - Output file for all functions of all variants with input test case 16 16 4194304

### CUDA
The results obtained on the CADE machine for each variant are stored respectively in each of the variants' folders, i.e., follow the steps to locate the folder with the CADE output files.

For example, if you want to see the output for all input cases of the variant ab :

Steps :
1. Enter the correct folder.
```
cd proj/Cuda/cudaab
```
2. See the *output* file.
```
cat output
```

You can enter any variant folder to see the corresponding output for all test cases.

## Lonepeak performance
Outputs obtained on the CHPC Lonepeak clusters are shown in the report file *HPC_Project_Report*. This file can be located in the root folder of this repo.

## Project Report
The Project report *HPC_Project_Report* is located in the root folder. The report contains output from the Lonepeak clusters, the CADE machine, and an explanation of each case.

