#pragma once

#include <vector>
#include <string>
#include "main.hpp"
#ifdef GPURUN
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <cuda_runtime.h>
#endif
using uint=unsigned int;

namespace my_lib
{
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//
   template <typename T> class vecP
   {
      public:
         T x[P];
#ifdef GPURUN
         __device__ __host__
#endif
         vecP(){
            for(int p = 0; p < P; p++){
               x[p] = 0;
            }
         }

#ifdef GPURUN
         __device__ __host__
#endif
         ~vecP(){};

#ifdef GPURUN
         __device__ __host__
#endif
         inline vecP& set_zero()
         {
            for(int p = 0; p < P; p++){
               this->x[p] = 0;
            }
            return *this;
         }

         template <typename T1>
#ifdef GPURUN
         __device__ __host__
#endif
         inline vecP& operator = (const T1& b)
         {
            for(int p = 0; p < P; p++){
               this->x[p] = b;
            }
            return *this;
         }

         template <typename T1>
#ifdef GPURUN
         __device__ __host__
#endif
         inline vecP& operator += (const T1& b)
         {
            for(int p = 0; p < P; p++){
               this->x[p] += b.x[p];
            }
            return *this;
         }

         template <typename T1>
#ifdef GPURUN
         __device__ __host__
#endif
         inline vecP& operator -= (const T1& b)
         {
            for(int p = 0; p < P; p++){
               this->x[p] -= b.x[p];
            }
            return *this;
         }

         template <typename T1>
#ifdef GPURUN
         __device__ __host__
#endif
         inline vecP& operator *= (const T1& b)
         {
            for(int p = 0; p < P; p++){
               this->x[p] *= b.x[p];
            }
            return *this;
         }
   };
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//

   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//
   template <typename T> class vecQ
   {
      public:
         T f[Q];
#ifdef GPURUN
         __device__ __host__
#endif
         vecQ(){
            for(int q = 0; q < Q; q++){
               f[q] = 0;
            }
         }
#ifdef GPURUN
         __device__ __host__
#endif
         ~vecQ(){};

#ifdef GPURUN
         __device__ __host__
#endif
         inline vecQ& set_zero()
         {
            for(int q = 0; q < Q; q++){
               this->f[q] = 0;
            }
            return *this;
         }

         template <typename T1>
#ifdef GPURUN
         __device__ __host__
#endif
         inline vecQ& operator += (const T1& b)
         {
            for(int q = 0; q < Q; q++){
               this->f[q] += b.f[q];
            }
            return *this;
         }

         template <typename T1>
#ifdef GPURUN
         __device__ __host__
#endif
         inline vecQ& operator -= (const T1& b)
         {
            for(int q = 0; q < Q; q++){
               this->f[q] -= b.f[q];
            }
            return *this;
         }
         template <typename T1>
#ifdef GPURUN
         __device__ __host__
#endif
         inline vecQ& operator *= (const T1& b)
         {
            for(int q = 0; q < Q; q++){
               this->f[q] *= b.f[q];
            }
            return *this;
         }
         template <typename T1>
#ifdef GPURUN
         __device__ __host__
#endif
         inline vecQ& operator /= (const T1& b)
         {
            for(int q = 0; q < Q; q++){
               this->f[q] /= b.f[q];
            }
            return *this;
         }
   };
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//
   
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//
   template <typename T> class vec3
   {
      public:
         T x[3];
#ifdef GPURUN
         __device__ __host__
#endif
         vec3(){
            for(int i = 0; i < 3; i++){
               x[i] = 0;
            }
         }

#ifdef GPURUN
         __device__ __host__
#endif
         ~vec3(){};

#ifdef GPURUN
         __device__ __host__
#endif
         inline vec3& set_zero()
         {
            for(int i = 0; i < 3; i++){
               this->x[i] = 0;
            }
            return *this;
         }

         template <typename T1>
#ifdef GPURUN
         __device__ __host__
#endif
         inline vec3& operator = (const T1& b)
         {
            for(int i = 0; i < 3; i++){
               this->x[i] = b;
            }
            return *this;
         }

         template <typename T1>
#ifdef GPURUN
         __device__ __host__
#endif
         inline vec3& operator += (const T1& b)
         {
            for(int i = 0; i < 3; i++){
               this->x[i] += b.x[i];
            }
            return *this;
         }

         template <typename T1>
#ifdef GPURUN
         __device__ __host__
#endif
         inline vec3& operator -= (const T1& b)
         {
            for(int i = 0; i < 3; i++){
               this->x[i] -= b.x[i];
            }
            return *this;
         }

         template <typename T1>
#ifdef GPURUN
         __device__ __host__
#endif
         inline vec3& operator *= (const T1& b)
         {
            for(int i = 0; i < 3; i++){
               this->x[i] *= b.x[i];
            }
            return *this;
         }
   };
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//

   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//
   template <typename T> struct Object
   {
      int        num_ele;
      int        num_node;
      vec3<int>* ele;
      vec3<T>*   node;

      int        num_line_ele;
      int        num_line_node;
      int*       line_ele;
      vec3<T>*   line_node;
      std::string name;
   };
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//

}

//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//
#include "slicer.hpp"
#include "input.hpp"
#include "output.hpp"
#include "malloc.hpp"
#include "vector_funcs.hpp"
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//
