#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "assert.h"
using namespace std;

void readFile(string filename, int* numbers)
{
    ifstream infile (filename);
    // vector<int> numbers;
    // int numbers[10000];
    string line;
    int index = 0;

    while(getline(infile, line))
    {
        stringstream ss (line);
        string sint;
        while(getline(ss, sint, ','))
        {
            // numbers.push_back(stoi(sint));
            numbers[index] = stoi(sint);
            index += 1;
        }
    }

    // return numbers;

}

__global__ void q2a_global_counter(int* gpu_out, int* gpu_in, int n)
{
    for(int i=0; i<n; i++)
    {
        int hundreds_value = gpu_in[i] / 100;
        if(hundreds_value == blockIdx.x)
        {
            gpu_out[hundreds_value] += 1;
        }
    }
}

__global__ void q2b_shared_mem_counter(int* gpu_out, int* gpu_in, int n)
{
    extern __shared__ int shared_in[];
    extern __shared__ int shared_out;
    int tidx = threadIdx.x;
    // int idx = tidx + blockIdx.x * blockDim.x;
    
    shared_out = 0;
    if(tidx == 0)
    {
        
        for(int i=0;i<n;i++)
        {
            shared_in[i] = gpu_in[i];
        }
    }
    __syncthreads(); // make sure entire block is loaded!


    for(int i=0;i<n;i++)
    {
        int hundreds_value = shared_in[i] / 100;
        if(hundreds_value == blockIdx.x)
            shared_out += 1;
    }
    __syncthreads();
    if(tidx == 0)
    {
        gpu_out[blockIdx.x] = shared_out;
    }

    __syncthreads();  
}


__global__ void q2c_prll_prfx_scan(int* gpu_out, int* gpu_in, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n)
    {
        return;
    }
    int total = 0;
    for (int i = 0; i <= idx; i++)
    {
        total += gpu_in[i];
    }
    gpu_out[idx] = total;
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


    int numbers[10000] = {}; //hardcoded size?
    readFile("inp.txt", numbers);
    // const int IN_SIZE = 10000;
    // const int IN_BYTES = numbers.size() * sizeof(int);
    const int IN_BYTES = sizeof(numbers);
    const int IN_SIZE = IN_BYTES / sizeof(numbers[0]);

    const int OUT_SIZE = 10;
    const int OUT_BYTES = OUT_SIZE * sizeof(int);

    // const int maxThreadsPerBlock = 512;
    // int threads = maxThreadsPerBlock;
    // int blocks = IN_SIZE / maxThreadsPerBlock;

    int blocks = 10;
    // int threads = IN_SIZE / blocks;
    int threads = 2;
    printf("blocks: %d   threads: %d\n\n", blocks, threads);


    int *gpu_in;
    int *gpu_out_2a;
    int *gpu_out_2b;
    int *gpu_out_2c;
    int cpu_out_2a[OUT_SIZE] = {0};
    int cpu_out_2b[OUT_SIZE] = {0};
    int cpu_out_2c[OUT_SIZE] = {0};

    cudaError_t ret;

    float elapsedTime_2a;
    float elapsedTime_2b;
    float elapsedTime_2c;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    

    ret = cudaMalloc((void **) &gpu_in, IN_BYTES);
    printf("gpu_in Malloc %s\n", ret == cudaSuccess ? "Success!": cudaGetErrorString(ret));
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
    cudaEventRecord(start, 0);
    q2a_global_counter<<<blocks, threads>>>(gpu_out_2a, gpu_in, IN_SIZE);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime_2a, start, stop);

    cudaEventRecord(start, 0);
    q2b_shared_mem_counter<<<blocks, threads, (IN_SIZE+OUT_SIZE)*sizeof(int)>>>(gpu_out_2b, gpu_in, IN_SIZE);    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime_2b, start, stop);

    cudaEventRecord(start, 0);
    q2c_prll_prfx_scan<<<blocks, threads>>>(gpu_out_2c, gpu_out_2a, IN_SIZE);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime_2c, start, stop);


    ret = cudaMemcpy(cpu_out_2a, gpu_out_2a, OUT_BYTES, cudaMemcpyDeviceToHost);
    printf("cpu_out_2a Memcpy %s\n", ret == cudaSuccess ? "Success!": cudaGetErrorString(ret));
    ret = cudaMemcpy(cpu_out_2b, gpu_out_2b, OUT_BYTES, cudaMemcpyDeviceToHost);
    printf("cpu_out_2b Memcpy %s\n", ret == cudaSuccess ? "Success!": cudaGetErrorString(ret));
    ret = cudaMemcpy(cpu_out_2c, gpu_out_2c, OUT_BYTES, cudaMemcpyDeviceToHost);
    printf("cpu_out_2c Memcpy %s\n", ret == cudaSuccess ? "Success!": cudaGetErrorString(ret));
    

    // correct output:
    // 2a & b): 510, 1095, 1051, 1035, 1063, 1012, 1067, 1053, 1053, 1061
    // 2c) 510, 1605, 2656, 3691, 4754, 5766, 6833, 7886, 8939, 10000
    printf("\n\n2a:   %f\n", elapsedTime_2a);
    for(int i=0;i<OUT_SIZE;i++)
    {
        printf("%d=%d  ", i, cpu_out_2a[i]);
    }

    printf("\n\n2b:   %f\n", elapsedTime_2b);
    for(int i=0;i<OUT_SIZE;i++)
    {
        printf("%d=%d  ", i, cpu_out_2b[i]);
    }


    printf("\n\n2c:   %f\n", elapsedTime_2c);
    for(int i=0;i<OUT_SIZE;i++)
    {
        printf("%d=%d  ", i, cpu_out_2c[i]);
    }

    printf("\n\n");

    cudaFree(gpu_in);
    cudaFree(gpu_out_2a);
    cudaFree(gpu_out_2b);
    cudaFree(gpu_out_2c);
    return 0;
}

