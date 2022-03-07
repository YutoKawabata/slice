#pragma once

#include <iostream>
#include <typeinfo>
#include "mydef.hpp"
#ifdef GPURUN
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <cuda_runtime.h>
#endif
#define TO_STRING(VariableName) # VariableName

namespace my_lib
{
   template <typename T>
   inline void MallocHost (T **f, unsigned int size)
   //===============================================================
   {
     if(size == 0){
        *f = NULL;
        return;
     }
     (*f) = (T *)calloc(size, sizeof(T)); 
     if (*f == NULL) {
       printf(" malloc host error! size=%d, typesize=%d\n", size, sizeof(T));
       std::string varname = TO_STRING(f);
       exit(1);
     }
   };
   //---------------------------------------------------------------
   //

#ifdef GPURUN
   template <typename T> 
   inline void MallocDeviceDP (T **f, unsigned int size)
   //===============================================================
   {
   //  checkCudaErrors(cudaMalloc((void**)f), sizeof(T)*size);
     cudaError_t errCode = cudaMalloc((void**)f, sizeof(T)*size);
     if (errCode != cudaSuccess) {
       printf(" malloc device error! %s\n", cudaGetErrorString(errCode));
       exit(1);
     }
   };
   //---------------------------------------------------------------
   template <typename T> 
   inline void MallocDevice (T **f, unsigned int size)
   //===============================================================
   {
     //checkCudaErrors(cudaMalloc((void**)f, sizeof(T)*size));
     if(size == 0){
        *f = NULL;
        return;
     }
     cudaError_t errCode = cudaMalloc((void**)f, sizeof(T)*size);
     if (errCode != cudaSuccess) {
       printf(" malloc device error! %s\n", cudaGetErrorString(errCode));
       exit(1);
     }
     checkCudaErrors(cudaMemset((*f), 0, sizeof(T)*size));
   };
   //---------------------------------------------------------------
   template <typename T>
   inline void MemcpyHtoD (T *f_dev, T *f_host, unsigned int size)
   //===============================================================
   {
     if(size == 0){
        f_dev = NULL;
        return;
     }
     checkCudaErrors(cudaMemcpy(f_dev, f_host, sizeof(T)*size, cudaMemcpyHostToDevice));
   }
   //---------------------------------------------------------------
   template <typename T>
   inline void MemcpyDtoH (T *f_dev, T *f_host, unsigned int size)
   //===============================================================
   {
     if(size == 0){
        f_host = NULL;
        return;
     }
     checkCudaErrors(cudaMemcpy(f_host, f_dev, sizeof(T)*size, cudaMemcpyDeviceToHost));
   }
   //---------------------------------------------------------------

   template <typename T>
   inline void MemcpyDtoD (T *f_to, T *f_from, unsigned int size)
   //===============================================================
   {
     if(size == 0){
        f_to = NULL;
        return;
     }
     checkCudaErrors(cudaMemcpy(f_to, f_from, sizeof(T)*size, cudaMemcpyDeviceToDevice));
   }
   //---------------------------------------------------------------
#endif
}
