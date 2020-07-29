///Note to Professor
//I hahve created two independant method for Qa part A and part B.
//Because part A update the input passed to the method. 
//There are some commented printf statement there in the file. Uncomment those in case to view intermediate values.
#include <stdio.h>
using namespace std;
#include <limits.h>
#include <cuda_runtime.h> 
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <stdlib.h>
#include <ctime>

const char FILENAME[] = "inp.txt";

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
__global__ void minParallel(int *DeviceInput, int *returnMinValue, int *DeviceLock,   int n)
{
   // printf("n %d, threadindex :%d, blockIdx:%d , blockDim:%d \n",	n,threadIdx.x ,blockIdx.x,blockDim.x);   
	int index = threadIdx.x + blockIdx.x*blockDim.x;
    int increment = 0;
  
    __shared__ int Container[2];    
    __syncthreads();

	int temp = INT_MAX;
	while(index + increment < n){
       // printf("\nindex:%d Compare temp %f with next DeviceInput index %d value %f  thread %d block %d \n",index,	temp, (index + increment),DeviceInput[index + increment], threadIdx.x ,blockIdx.x);  
		temp = min(temp, DeviceInput[index + increment]);
        increment += blockDim.x;
	} 
    //printf("\nindex:%d temp is %f thread %d block %d \n",index,	temp,threadIdx.x ,blockIdx.x);   
	Container[threadIdx.x] = temp;

    __syncthreads(); 
    //Uncomment below for printing initial value of COntainer
    // if(index == 0){
    //       printf("\nvalue in intial Container is :");	
    //     for(unsigned int i=0;i<blockDim.x;i++){
    //         printf(" %d", 	Container[i]);  
        
    //         }
    //         printf("\n");
    //     }
    //     __syncthreads();
	// reduce the array by divide into 2
    unsigned int i = blockDim.x/2;
	while(i != 0){
        //printf("\ncomparing threadIdx.x %d %d and i %d ",threadIdx.x , i);	
		if(threadIdx.x < i ){ 
            
            Container[threadIdx.x] = min(Container[threadIdx.x], Container[threadIdx.x + i]);
            if(  index == 0 && i % 2 ==1){ //Special conition for handling last odd number 
             Container[index] = min(Container[index], Container[i-1]);

            }

		}

        __syncthreads();
       
		i /= 2;
    }
    __syncthreads(); 
    //Printing final value in COntainer
    // if(index == 0){
    //      // printf("\nvalue in final Container is :");	
    //     for(unsigned int i=0;i<blockDim.x;i++){
    //         printf(" %d", Container[i]);  
        
    //         }
    //         printf("\n");
    //     }
 


    __syncthreads();   
        if(index == 0){
        while(atomicCAS(DeviceLock,0,1) != 0);  //Lock the maxvalue
    //printf("*returnMinValue is %d cahce[0] is %d cahce[1] is %d\n",*returnMinValue,Container[0] ,Container[1]);
        *returnMinValue =   min(Container[0],Container[1]);
		atomicExch(DeviceLock, 0);  //unlock the max value
    }
    __syncthreads(); //New
} 
__global__ void calculateLastDigit(int *DeviceInput, int *DeviceOutput,     int n)
{
   int index = threadIdx.x + blockIdx.x*blockDim.x;
    int increment = 0;
	while(index + increment < n){
        DeviceOutput[index + increment ]=DeviceInput[index + increment ]%10;
        increment += gridDim.x*blockDim.x;;
	} 
 
    __syncthreads(); 
    
}



void maina()
{
    printf("Entering Q1A.....\n");
    cudaError_t ret;
    //int SelectLimit =348; //This will limit the number of element to fetch from file. Only used if we use readFile function.
 
	// Maxvalue in Host
    int *host_MaxVal; 
    host_MaxVal = (int*)malloc(sizeof(int));

    //input data in Host
    int *HostInput;
    //HostInput = (int*)malloc(SelectLimit*sizeof(int));

    

    //Mutex data in Device
    int *device_mutex;
    cudaMalloc((void**)&device_mutex, sizeof(int));
    cudaMemset(device_mutex, 0, sizeof(int));

    int *device_maxValue;
    cudaMalloc((void**)&device_maxValue, sizeof(int));    
    cudaMemset(device_maxValue, INT_MAX, sizeof(int));

 

    //Read from flat file if we need to limit selection
    //readFile("inp.txt", HostInput, SelectLimit);
    //readFileAll("inp.txt", HostInput,flatFileCount);

    /////////////////////Start Reading File
                            ifstream infile (FILENAME);
                            string line;
                            int index = 0;
                            int Linecount=0;
                            while(getline(infile, line))
                            {
                            stringstream ss (line);
                            string sint;
                            while(getline(ss, sint, ','))
                            {
                                Linecount++;
                            }
                            } 
                            printf("Number of Elemnts in file is %d\n",Linecount);
                            //Input data in Device
                            int *DeviceInput;
                            cudaMalloc((void**)&DeviceInput, Linecount*sizeof(int));
                            HostInput = (int*)malloc(Linecount*sizeof(int));
                            ifstream inReadfile (FILENAME);
                            //numbers = (int*)malloc((Linecount)*sizeof(int));
                            string Newline;
                            while(getline(inReadfile, Newline))
                            {
                            stringstream ss (Newline);
                            string sint;
                            while(getline(ss, sint, ','))
                            {
                                HostInput[index] = stoi(sint); 
                                index++;
                            }
                            } 
                            //Uncomment to see the elemnts checking
                            // printf("Element in file host :");
                            // for(int i=0;i<Linecount;i++){
                            // printf(" %d",	 HostInput[i] );  
                            // }
                            // printf("\n");
    /////////////////////End Reading file
 
	cudaMemcpy(DeviceInput, HostInput, Linecount*sizeof(int), cudaMemcpyHostToDevice);
    minParallel<<< 10, 10 >>>(DeviceInput, device_maxValue, device_mutex, Linecount);

	// copy from device to host
    ret = cudaMemcpy(host_MaxVal, device_maxValue, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Returning Max Value from Device is :  %s\n", ret == cudaSuccess ? "Success!": cudaGetErrorString(ret)); 

	//report results
	printf("Parallel Minimum: %d\n",*host_MaxVal);
 
    int* resultA;
    resultA = (int*)malloc(1 * sizeof(int));
    resultA[0]=*host_MaxVal;
    writeFile("q1a.txt", resultA,1);

	// Check Sequential Execution and compare the result
	clock_t cpu_start = clock();
	for(  int j=0;j<1000;j++){
		*host_MaxVal = INT_MAX;
		for(  int i=0;i<Linecount;i++){
			if(HostInput[i] < *host_MaxVal){
				*host_MaxVal = HostInput[i];
			}
		}
	}
	printf("Sequential Minimum : %d\n",*host_MaxVal);

	//Release all the elemnts in the device and host
	free(HostInput);
	free(host_MaxVal);
	cudaFree(DeviceInput);
	cudaFree(device_maxValue);
    cudaFree(device_mutex);
    printf("Finished Q1A.....\n\n\n\n");
}
void mainb()
{
    printf("Entering Q1B.....\n");
    cudaError_t ret;
 
    int *HostInput;
   

    // /////////////////////Start Reading File
                            printf("Staring reading file\n");
                            ifstream infile(FILENAME);
                            string line;
                            int index = 0;
                            int Linecount=0;
                            while(getline(infile, line))
                            {
                            stringstream ss (line);
                            string sint;
                            while(getline(ss, sint, ','))
                            {
                                Linecount++;
                            }
                            } 
                            printf("Number of Elemnts in file is %d\n",Linecount);
                            //Input data in Device
                            int *DeviceInput;
                            cudaMalloc((void**)&DeviceInput, Linecount*sizeof(int));
                             HostInput = (int*)malloc(Linecount*sizeof(int));
                             ifstream inReadfile (FILENAME);
                            string Newline;
                            while(getline(inReadfile, Newline))
                            {
                                stringstream ss (Newline);
                                string sint;
                                while(getline(ss, sint, ','))
                                {
                                    HostInput[index] = stoi(sint); 
                                    index++;
                                }
                            } 
                            //                         //Uncomment to see the elemnts checking
                                                    printf("Element in file host :");
                                                    for(int i=0;i<Linecount;i++){
                                                    printf(" %d",	 HostInput[i] );  
                                                    }
                                                    printf("\n");
                            // /////////////////////End Reading file
 
            int *DeviceOutput;
            ret = cudaMalloc((void **) &DeviceOutput, Linecount*sizeof(int));
            printf("Allocateing  DeviceOutput is %s\n", ret == cudaSuccess ? "Success!": cudaGetErrorString(ret));

            // int HostOutput[Linecount] = {0};
            int * HostOutput = (int *)calloc(Linecount, sizeof(int));
            ret = cudaMemcpy( DeviceInput,HostInput, Linecount*sizeof(int), cudaMemcpyHostToDevice);
            printf("Copying  DeviceOutput is  %s\n", ret == cudaSuccess ? "Success!": cudaGetErrorString(ret));
 
             calculateLastDigit<<< 2, 5 >>>(DeviceInput,DeviceOutput,  Linecount);

             ret = cudaMemcpy(HostOutput, DeviceOutput, Linecount*sizeof(int), cudaMemcpyDeviceToHost);
             printf("Copying back from Device for  DeviceOutput is %s\n", ret == cudaSuccess ? "Success!": cudaGetErrorString(ret));

           //  char *str="";
            for(  int i=0;i<Linecount;i++){
                    printf("Index:%d  Host Value:%d Last Digit:%d\n",i,HostInput[i],HostOutput[i] );
                //    *str+=HostOutput[i] +", ";
                }

                writeFile("q1b.txt", HostOutput,Linecount);
              //  writeFile("q1b.txt",*str);
  

	//Release all the elemnts in the device and host
	cudaFree(DeviceInput);
	cudaFree(DeviceOutput);
    printf("Finished Q1B.....\n\n\n\n\n");
}

int main()
{
 maina();
 mainb();
}