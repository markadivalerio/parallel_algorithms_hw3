#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <math.h>
#include "assert.h"

using namespace std;


std::vector<int> readFile(string filename)
{
    ifstream infile (filename);
    vector<int> vnum;
    string line;
    int index = 0;

    while(getline(infile, line))
    {
        stringstream ss (line);
        string sint;
        while(getline(ss, sint, ','))
        {
            vnum.push_back(stoi(sint));
            index += 1;
        }
    }
    printf("File size is %d",index);
    return vnum;
}


void writeFile(const char* filename, int* data)
{
    ofstream myfile (filename);
    if(myfile.is_open())
    {
        int size = sizeof(data);
        printf("size is %d",size);
        for(int i=0; i<=size+1; i++)
        {
            myfile << data[i];
            if(i<=size)
                myfile << ", ";
        }
        myfile.close();
    }
    else
    {
        printf("Unable to open/write to file %s", filename);
    }

}

__global__ void q1a_min(int* gpu_out, int* gpu_in, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int min = INT_MAX;
    for(int i=0; i<n;i++)
    {
        if(gpu_in[i] < min)
            min = gpu_in[i];
    }
    if(idx == 0)
        gpu_out[0] = min;
}

__global__ void q1b_ones_digit(int* gpu_out, int* gpu_in, int n)
{
    for(int i=0; i<n;i++)
    {
        gpu_out[i] = gpu_in[i] % 10;
    }
}

 
__global__ void q2b_shared_mem_counter(int* gpu_out, int* gpu_in, int n)
{
   // printf("gri dimention is %d \n",gridDim.x);
   // printf("Size inside CUDA is %d thread %d block %d ",n,threadIdx.x , blockIdx.x );
    extern __shared__ int shared_in[];
    extern __shared__ int shared_out[10];
    int tidx = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int numbersPerBlockThread = (n / (blockDim.x * gridDim.x)) + 1;
    int start = idx * numbersPerBlockThread;
    int stop = min(n, (start + numbersPerBlockThread));
   int  incrementadd=0;
   int temp=0;
//    if(tidx == 0)
//       {
    while((idx + incrementadd) < n && incrementadd < n){
       // int nDigits = floor(log10(abs(gpu_in[idx + incrementadd]))) + 1;
       if(gpu_in[idx + incrementadd]<100)
       {
        atomicAdd(&(shared_in[0]), 1);
       }
       else if(gpu_in[idx + incrementadd]>=100 && gpu_in[idx + incrementadd]<200)
       {
        atomicAdd(&(shared_in[1]), 1);
       }
       else if(gpu_in[idx + incrementadd]>=200 && gpu_in[idx + incrementadd]<300)
       {
        atomicAdd(&(shared_in[2]), 1);
       }   
       else if(gpu_in[idx + incrementadd]>=300 && gpu_in[idx + incrementadd]<400)
       {
        atomicAdd(&(shared_in[3]), 1);
       } 
       else if(gpu_in[idx + incrementadd]>=400 && gpu_in[idx + incrementadd]<500)
       {
        atomicAdd(&(shared_in[4]), 1);
       } 
       else if(gpu_in[idx + incrementadd]>=500 && gpu_in[idx + incrementadd]<600)
       {
        atomicAdd(&(shared_in[5]), 1);
       } 
       else if(gpu_in[idx + incrementadd]>=600 && gpu_in[idx + incrementadd]<700)
       {
        atomicAdd(&(shared_in[6]), 1);
       } 
       else if(gpu_in[idx + incrementadd]>=700 && gpu_in[idx + incrementadd]<800)
       {
        atomicAdd(&(shared_in[7]), 1);
       } 
       else if(gpu_in[idx + incrementadd]>=800 && gpu_in[idx + incrementadd]<900)
       {
        atomicAdd(&(shared_in[8]), 1);
       }
       else if(gpu_in[idx + incrementadd]>=900 && gpu_in[idx + incrementadd]<1000)
       {
        atomicAdd(&(shared_in[9]), 1);
       }
 
        incrementadd +=   (blockIdx.x+1) * blockDim.x;
       // printf("Increment %d",(idx + incrementadd));
    }
    __syncthreads();
   if(idx == 0)
     {
    //printf("Shared in count for less than 2 %d",shared_in[2]);
    for(int i=0;i<10;i++)
        {
            atomicAdd(&(gpu_out[i]), shared_in[i]);
           // gpu_out[i]=   shared_in[i]  ;
        }
  }
 
    __syncthreads();
    //printf("Completed thread %d block %d \n",threadIdx.x , blockIdx.x );
}

 


int main(int argc, char **argv)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp devProps;
    if (cudaGetDeviceProperties(&devProps, dev) == 0)
    {
        printf("Using device %d: %s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
               dev, devProps.name, (int)devProps.totalGlobalMem,
               (int)devProps.major, (int)devProps.minor,
               (int)devProps.clockRate);
    }

    vector<int> vnum = readFile("inp.txt");
    const int IN_SIZE = vnum.size();
    const int IN_BYTES = IN_SIZE * sizeof(int);
    const int OUT_SIZE = 10; //this is specific to the output range.
    const int OUT_BYTES = OUT_SIZE * sizeof(int);

    int* numbers;
    numbers = (int *)malloc(IN_BYTES);
    for(int i=0; i < vnum.size(); i++)
        numbers[i] = vnum[i];

    // const int maxThreadsPerBlock = 512;
    int blocks = 10;
    int threads = 10;
    printf("Input size: %d   blocks: %d   threads: %d\n\n", IN_SIZE, blocks, threads);


    int *gpu_in;
    int *gpu_out_1a;
    int *gpu_out_1b;
    int *gpu_out_2a;
    int *gpu_out_2b;
    int *gpu_out_2c;

    int cpu_out_1a[OUT_SIZE] = {0};
    int* cpu_out_1b;
    cpu_out_1b = (int *)calloc(IN_SIZE, sizeof(int));
    int cpu_out_2a[OUT_SIZE] = {0};
    int cpu_out_2b[OUT_SIZE] = {0};
    int cpu_out_2c[OUT_SIZE] = {0};

    cudaError_t ret;

    float elapsedTime_1a;
    float elapsedTime_1b;
    float elapsedTime_2a;
    float elapsedTime_2b;
    float elapsedTime_2c;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    

    ret = cudaMalloc((void **) &gpu_in, IN_BYTES);
    printf("gpu_in Malloc %s\n", ret == cudaSuccess ? "Success!": cudaGetErrorString(ret));
    ret = cudaMalloc((void **) &gpu_out_1a, IN_BYTES);
    printf("gpu_out_1a Malloc %s\n", ret == cudaSuccess ? "Success!": cudaGetErrorString(ret));
    ret = cudaMalloc((void **) &gpu_out_1b, IN_BYTES);
    printf("gpu_out_1b Malloc %s\n", ret == cudaSuccess ? "Success!": cudaGetErrorString(ret));
    ret = cudaMalloc((void **) &gpu_out_2a, OUT_BYTES);
    printf("gpu_out_2a Malloc %s\n", ret == cudaSuccess ? "Success!": cudaGetErrorString(ret));
    ret = cudaMalloc((void **) &gpu_out_2b, OUT_BYTES);
    printf("gpu_out_2b Malloc %s\n", ret == cudaSuccess ? "Success!": cudaGetErrorString(ret));
    ret = cudaMalloc((void **) &gpu_out_2c, OUT_BYTES);
    printf("gpu_out_2c Malloc %s\n", ret == cudaSuccess ? "Success!": cudaGetErrorString(ret));

    ret = cudaMemcpy((void *)gpu_in, (void *)numbers, IN_BYTES , cudaMemcpyHostToDevice);
    printf("gpu_in Memcpy %s\n", ret == cudaSuccess ? "Success!": cudaGetErrorString(ret));


    // see https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration 
    // for <<<Dg, Db, Ns, S>>> parameter explanation.

    // cudaEventRecord(start, 0);
    // q1a_min<<<blocks, threads>>>(gpu_out_1a, gpu_in, IN_SIZE);
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsedTime_1a, start, stop);


    // cudaEventRecord(start, 0);
    // q1b_ones_digit<<<blocks, threads>>>(gpu_out_1b, gpu_in, IN_SIZE);
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsedTime_1b, start, stop);


    // cudaEventRecord(start, 0);
    // q2a_global_counter<<<blocks, threads>>>(gpu_out_2a, gpu_in, IN_SIZE);
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsedTime_2a, start, stop);
printf("In_SIZE is %d\n",IN_BYTES);
printf("shared memory size to pass is %d\n", (IN_SIZE+OUT_SIZE)*sizeof(int));
    cudaEventRecord(start, 0);
   //q2b_shared_mem_counter<<<10, 10, (IN_SIZE+OUT_SIZE)*sizeof(int)>>>(gpu_out_2b, gpu_in, IN_SIZE);    
    q2b_shared_mem_counter<<<10, 200, 10000>>>(gpu_out_2b, gpu_in, IN_SIZE); 
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime_2b, start, stop);


    // cudaEventRecord(start, 0);
    // int n2 = (int)pow(2.0, ceil(log((double)OUT_SIZE)/log(2.0))); // next power of 2
    // q2c_prll_prfx_sum<<<blocks, threads, (n2)*sizeof(int)>>>(gpu_out_2c, gpu_out_2a, OUT_SIZE);
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsedTime_2c, start, stop);


    // ret = cudaMemcpy(cpu_out_1a, gpu_out_1a, OUT_BYTES, cudaMemcpyDeviceToHost);
    // printf("cpu_out_1a Memcpy %s\n", ret == cudaSuccess ? "Success!": cudaGetErrorString(ret));
    // ret = cudaMemcpy(cpu_out_1b, gpu_out_1b, IN_BYTES, cudaMemcpyDeviceToHost);
    // printf("cpu_out_1b Memcpy %s\n", ret == cudaSuccess ? "Success!": cudaGetErrorString(ret));
    // ret = cudaMemcpy(cpu_out_2a, gpu_out_2a, OUT_BYTES, cudaMemcpyDeviceToHost);
    // printf("cpu_out_2a Memcpy %s\n", ret == cudaSuccess ? "Success!": cudaGetErrorString(ret));
    ret = cudaMemcpy(cpu_out_2b, gpu_out_2b, OUT_BYTES, cudaMemcpyDeviceToHost);
    printf("cpu_out_2b Memcpy %s\n", ret == cudaSuccess ? "Success!": cudaGetErrorString(ret));
    // ret = cudaMemcpy(cpu_out_2c, gpu_out_2c, OUT_BYTES, cudaMemcpyDeviceToHost);
    // printf("cpu_out_2c Memcpy %s\n", ret == cudaSuccess ? "Success!": cudaGetErrorString(ret));
    

    // correct output:
    // 2a & b): 510, 1095, 1051, 1035, 1063, 1012, 1067, 1053, 1053, 1061
    // 2c) 510, 1605, 2656, 3691, 4754, 5766, 6833, 7886, 8939, 10000


    // printf("\n\n1a:   %f\n", elapsedTime_1a);
    // printf("Minimum = %d", cpu_out_1a[0]);



    // int correct_count = 0;
    // printf("\n\n1b:   %f\n", elapsedTime_1b);
    // for(int i=0;i<IN_SIZE;i++)
    // {
    //     // printf("%d ", cpu_out_1b[i]);
    //     if(numbers[i] % 10 == cpu_out_1b[i])
    //     {
    //         correct_count += 1;
    //     //     // printf("%d vs %d\n", numbers[i], cpu_out_1b[i]);
    //     }
    // }
    // printf("\nCount with correct ones digit:   %d / %d", correct_count, IN_SIZE);

    // printf("\n\n2a:   %f\n", elapsedTime_2a);
    // for(int i=0;i<OUT_SIZE;i++)
    // {
    //     printf("%d=%d  ", i, cpu_out_2a[i]);
    // }
    // writeFile("q2a.txt", cpu_out_2a);

    // printf("\n\n2b:   %f\n", elapsedTime_2b);
    for(int i=0;i<OUT_SIZE;i++)
    {
        printf("%d  ", cpu_out_2b[i]);
    }
    writeFile("q2b.txt", cpu_out_2b);

    // printf("\n\n2c:   %f\n", elapsedTime_2c);
    // for(int i=0;i<OUT_SIZE;i++)
    // {
    //     printf("%d=%d  ", i, cpu_out_2c[i]);
    // }
    // writeFile("q2c.txt", cpu_out_2c);

    printf("\n\n");

    cudaFree(gpu_in);
    cudaFree(gpu_out_1a);
    cudaFree(gpu_out_1b);
    cudaFree(gpu_out_2a);
    cudaFree(gpu_out_2b);
    cudaFree(gpu_out_2c);
    return 0;
}

