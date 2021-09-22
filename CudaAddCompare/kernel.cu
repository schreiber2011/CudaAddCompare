
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <chrono>
#include <iostream>
#include <ctime>

using std::cout; using std::endl;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size, unsigned int loop);

__global__ void addKernel(int* c, const int* a, const int* b, unsigned int loop)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
    for (size_t j = 0; j < loop; j++)
    {
        c[i]++;
    }
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };
    unsigned int loop = 100 * 1000;

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize, loop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size, unsigned int loop)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
    // Choose ... END

    // Allocate GPU buffers for three vectors (two input, one output)    .
    auto millisec_since_epoch = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    auto dif_millisec_since_epoch = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - millisec_since_epoch;
    cout << "Malloc time in milliseconds: " << dif_millisec_since_epoch << endl;
    // End of allocation

    // Copy input vectors from host memory to GPU buffers.
    millisec_since_epoch = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    dif_millisec_since_epoch = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - millisec_since_epoch;
    cout << "Copy time in milliseconds: " << dif_millisec_since_epoch << endl;
    // Copy ... END

        /******************************************************************
                                       GPU sum
            Launch a kernel on the GPU with one thread for each element.

        ******************************************************************/
    for (size_t i = 1; i < 200; i = i * 10)
    {
        millisec_since_epoch = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
        cout << "Loop count: " << loop * i << endl;
        addKernel << <1, size >> > (dev_c, dev_a, dev_b, loop * i);
        cudaDeviceSynchronize();
        dif_millisec_since_epoch = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - millisec_since_epoch;
        cout << "Add time in milliseconds iterations " << loop * i << ": " << dif_millisec_since_epoch << endl;
    }

    // TODO: Divide task into smaller chunks
    /*
    for (size_t i = 1; i < 200; i = i * 10)
    {
        millisec_since_epoch = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
        cout << loop * i << endl;
        addKernel << <1, size >> > (dev_c, dev_a, dev_b, loop * i);
        cudaDeviceSynchronize();
        dif_millisec_since_epoch = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - millisec_since_epoch;
        cout << "Add time in milliseconds iterations " << loop * i << ": " << dif_millisec_since_epoch << endl;
    }
    */
    // TODO (in progress) ... END


    /**********************************************
                          CPU sum
    **********************************************/
    int cpu_mutiplier = 10;
    for (size_t i = cpu_mutiplier; i < 200 * cpu_mutiplier; i = i * 10)
    {
        millisec_since_epoch = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
        cout << "Loop count: " << loop * i << ", size of array: " << size << endl;
        for (size_t k = 0; k < size; k++)
        {
            c[k] = a[k] + b[k];
            for (size_t j = 0; j < (loop * i); j++)
            {
                c[k]++;
            }
        }
        dif_millisec_since_epoch = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - millisec_since_epoch;
        cout << "Add (CPU) time in milliseconds iterations " << loop * i << ": " << dif_millisec_since_epoch << endl;
    }
    // END ... cpu sum

// Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}
