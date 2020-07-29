#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
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

    return vnum;
}


void writeFile(const char* filename, int* data, int limit)
{
    ofstream myfile(filename);
    if (myfile.is_open())
    {
        for (int i = 0; i < limit; i++)
        {
            myfile << data[i];
         if(i!= limit-1)
            myfile << ", ";
        }
        myfile.close();
    }
    else
    {
        printf("Unable to open/write to file %s", filename);
    }

}

__global__ void q2a_global_counter(int* gpu_out, int* gpu_in, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int numbersPerBlockThread = (n / (blockDim.x * gridDim.x)) + 1;
    int start = idx * numbersPerBlockThread;
    int stop = min(n, (start + numbersPerBlockThread));

    for(int i=start; i<stop; i++)
    {
        int hundreds_value = gpu_in[i] / 100;
        atomicAdd(&(gpu_out[hundreds_value]), 1);
    }
    __syncthreads();
}

__global__ void q2b_shared_mem_counter(int* gpu_out, int* gpu_in, int n)
{
    extern __shared__ int shared_out[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int incrementalAdd = 0;
    if(threadIdx.x == 0)
    {
        shared_out[10] = {0};
    }

    while(idx + incrementalAdd < n)
    {
        int hundreds_val = gpu_in[idx + incrementalAdd] / 100;
        atomicAdd(&shared_out[hundreds_val], 1);
        incrementalAdd += (blockIdx.x+1) * blockDim.x;
    }
    __syncthreads();

    if(idx == 0)
    {
        for(int i=0; i < 10; i++)
        {
            atomicAdd(&(gpu_out[i]), shared_out[i]);
            __syncthreads();
        }
    }
}


__global__ void q2c_prll_prfx_sum(int* gpu_out, int* gpu_in, int n)
{
    // gpu_in = [510,1095,1051,1035,1063,1012,1067,1053,1053,1061]
    // gpu_out =[510, 1605, 2656, 3691, 4754, 5766, 6833, 7886, 8939, 10000]
    extern __shared__ int shared_mem[];

    int tidx = threadIdx.x;
    int idx = tidx + blockIdx.x * blockDim.x;;
    int offset = 1;
    int n2 = (int)pow(2.0, ceil(log((double)n)/log(2.0))); // next power of 2

    if(tidx == 0)
    {
        for(int i=0;i<n2;i++)
        {
            if(i<n)
                shared_mem[i] = gpu_in[i];
            else
                shared_mem[i] = 0; // extend non-power-of-2 entries to 0.
        }
    }
    __syncthreads();

    // UPWARD FIRST PASS SUM

    for(int depth_idx = n2 >> 1; depth_idx > 0; depth_idx >>= 1)
    {
        __syncthreads();

        if(tidx < depth_idx)
        {
            int ai = offset*(2*tidx+1)-1;
            int bi = offset*(2*tidx+2)-1;
            shared_mem[bi] += shared_mem[ai];
        }
        offset <<= 1;
    }

    __syncthreads();

    // UPWARD FIRST PASS ENDS AND 2nd DOWNWARD PASS BEGINS

    if(tidx == 0)
    {
        shared_mem[n2-1] = 0;
    }

    for(int depth_idx=1; depth_idx<n2; depth_idx<<=1)
    {
        offset >>= 1;
        __syncthreads();
        if(tidx < depth_idx)
        {
            int ai = offset*(2*tidx+1)-1;
            int bi = offset*(2*tidx+2)-1;
            if(ai < 0 || bi < 0)
                continue;

            int temp = shared_mem[ai];
            shared_mem[ai] = shared_mem[bi];
            shared_mem[bi] += temp;
        }
    }
    __syncthreads();

    if(idx < n)
    {
        gpu_out[idx] = shared_mem[idx+1];
    }
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
        printf("Using device %d: %s\nglobal mem: %dB; compute v%d.%d; clock: %d kHz\n",
               dev, devProps.name, (int)devProps.totalGlobalMem,
               (int)devProps.major, (int)devProps.minor,
               (int)devProps.clockRate);
        printf("sharedMemPerBlock: %zu    sharedMemPerMultiprocessor: %zu\n", devProps.sharedMemPerBlock, devProps.sharedMemPerMultiprocessor);
        printf("regsPerMultiprocessor:  %d\n", devProps.regsPerMultiprocessor);
    }

    vector<int> vnum = readFile("inp.txt");
    // vector<int> vnum = readFile("inp1mil.txt");
    const int IN_SIZE = vnum.size();
    const int IN_BYTES = IN_SIZE * sizeof(int);
    const int OUT_SIZE = 10; //this is specific to the output range.
    const int OUT_BYTES = OUT_SIZE * sizeof(int);

    int* numbers;
    numbers = (int *)malloc(IN_BYTES);
    for(int i=0; i < vnum.size(); i++)
        numbers[i] = vnum[i];

    // int MAX_THREADS_PER_BLOCK  = 512;
    int threads = 32;
    int blocks = 8;

    printf("Input size: %d   blocks: %d   threads: %d\n\n", IN_SIZE, blocks, threads);

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
    q2b_shared_mem_counter<<<blocks, threads, OUT_SIZE*sizeof(int)>>>(gpu_out_2b, gpu_in, IN_SIZE);
    // ret = cudaPeekAtLastError();
    // printf("cudaPeekAtLastError %s\n", ret == cudaSuccess ? "Success!": cudaGetErrorString(ret));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime_2b, start, stop);


    cudaEventRecord(start, 0);
    int n2 = (int)pow(2.0, ceil(log((double)OUT_SIZE)/log(2.0))); // next power of 2
    q2c_prll_prfx_sum<<<blocks, threads, (n2)*sizeof(int)>>>(gpu_out_2c, gpu_out_2a, OUT_SIZE);
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
    writeFile("q2a.txt", cpu_out_2a, OUT_SIZE);

    printf("\n\n2b:   %f\n", elapsedTime_2b);
    for(int i=0;i<OUT_SIZE;i++)
    {
        printf("%d=%d  ", i, cpu_out_2b[i]);
    }
    writeFile("q2b.txt", cpu_out_2b, OUT_SIZE);

    printf("\n\n2c:   %f\n", elapsedTime_2c);
    for(int i=0;i<OUT_SIZE;i++)
    {
        printf("%d=%d  ", i, cpu_out_2c[i]);
    }
    writeFile("q2c.txt", cpu_out_2c, OUT_SIZE);

    printf("\n\n");

    cudaFree(gpu_in);
    cudaFree(gpu_out_2a);
    cudaFree(gpu_out_2b);
    cudaFree(gpu_out_2c);
    return 0;
}

