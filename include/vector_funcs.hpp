#pragma once

#include "mydef.hpp"

namespace my_lib
{
   /*--- vecP ---*/
   template <typename T>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline T dot(vecP<T> a, vecP<T> b)
   {
      T c = 0;
      for(int p = 0; p < P; p++){
         c += a.x[p]*b.x[p];
      }
      return c;
   }
   //================================================================================

   template <typename T>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline T dot(vecP<T> a, vecP<int> b)
   {
      T c = 0;
      for(int p = 0; p < P; p++){
         c += a.x[p]*b.x[p];
      }
      return c;
   }
   //================================================================================

   template <typename T>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline vecP<T> cross(vecP<T> a, vecP<T> b)
   {
      vecP<T> c;
      c.x[0] = a.x[1]*b.x[2] - a.x[2]*b.x[1];
      c.x[1] = a.x[2]*b.x[0] - a.x[0]*b.x[2];
      c.x[2] = a.x[0]*b.x[1] - a.x[1]*b.x[0];
      return c;
   }
   //================================================================================
   
   template <typename T>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline T norm(vecP<T> a)
   {
      T c = sqrt(dot<T>(a, a));
      return c;
   }
   //================================================================================

   template <typename T>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline T norm2(vecP<T> a)
   {
      return dot<T>(a,a);
   }
   //================================================================================

   template <typename T, typename U>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline vecP<T> operator + (vecP<T> a, U b)
   {
     vecP<T> c;
     for(int p = 0; p < P; p++){
        c.x[p] = a.x[p] + b;
     }
     return c;
   }
   //================================================================================

   template <typename T, typename U>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline vecP<T> operator - (vecP<T> a, U b)
   {
     vecP<T> c;
     for(int p = 0; p < P; p++){
        c.x[p] = a.x[p] - b;
     }
     return c;
   }
   //================================================================================

   template <typename T, typename U>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline vecP<T> operator * (vecP<T> a, U b)
   {
     vecP<T> c;
     for(int p = 0; p < P; p++){
        c.x[p] = a.x[p] * b;
     }
     return c;
   }
   //================================================================================

   template <typename T, typename U>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline vecP<T> operator / (vecP<T> a, U b)
   {
     vecP<T> c;
     for(int p = 0; p < P; p++){
        c.x[p] = a.x[p] / b;
     }
     return c;
   }
   //================================================================================

   template<typename T, typename U>
#ifdef GPURUN
   __device__ __host__
#endif
   inline vecP<U> vecP_cast(vecP<T> a)
   {
     vecP<U> c;
     for(int p = 0; p < P; p++){
        c.x[p] = static_cast<U>(a.x[p]);
     }
     return c;
   }
   //================================================================================

   template <typename T>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline vecP<T> operator + (vecP<T> a, vecP<T> b)
   {
     vecP<T> c;
     for(int p = 0; p < P; p++){
        c.x[p] = a.x[p] + b.x[p];
     }
     return c;
   }
   //================================================================================

   template <typename T>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline vecP<T> operator - (vecP<T> a, vecP<T> b)
   {
     vecP<T> c;
     for(int p = 0; p < P; p++){
        c.x[p] = a.x[p] - b.x[p];
     }
     return c;
   }
   //================================================================================

   template <typename T>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline vecP<T> operator * (vecP<T> a, vecP<T> b)
   {
     vecP<T> c;
     for(int p = 0; p < P; p++){
        c.x[p] = a.x[p] * b.x[p];
     }
     return c;
   }
   //================================================================================

   template <typename T>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline vecP<T> operator / (vecP<T> a, vecP<T> b)
   {
     vecP<T> c;
     for(int p = 0; p < P; p++){
        c.x[p] = a.x[p] / b.x[p];
     }
     return c;
   }
   //================================================================================

   template <typename T>
#ifdef GPURUN
   __device__ __host__
#endif
   inline bool operator == (vecP<T> a, vecP<T> b)
   {
      bool flag = true;
      for(int p = 0; p < P; p++){
         if(std::abs(a.x[p] - b.x[p]) > TOL){
            flag = false;
            break;
         }
      }
      return flag;
   }
   //================================================================================

   template <typename T>
#ifdef GPURUN
   __device__ __host__
#endif
   inline bool operator < (vecP<T> a, vecP<T> b)
   {
      bool flag = true;
      for(int p = 0; p < P; p++){
         if(a.x[p] >= b.x[p]){
            flag = false;
            break;
         }
      }
      return flag;
   }
   //================================================================================

   template <typename T>
#ifdef GPURUN
   __device__ __host__
#endif
   inline bool operator > (vecP<T> a, vecP<T> b)
   {
      bool flag = true;
      for(int p = 0; p < P; p++){
         if(a.x[p] <= b.x[p]){
            flag = false;
            break;
         }
      }
      return flag;
   }
   //================================================================================

   template <typename T>
#ifdef GPURUN
   __device__ __host__
#endif
   inline bool operator <= (vecP<T> a, vecP<T> b)
   {
      bool flag = true;
      for(int p = 0; p < P; p++){
         if(a.x[p] > b.x[p]){
            flag = false;
            break;
         }
      }
      return flag;
   }
   //================================================================================

   template <typename T>
#ifdef GPURUN
   __device__ __host__
#endif
   inline bool operator >= (vecP<T> a, vecP<T> b)
   {
      bool flag = true;
      for(int p = 0; p < P; p++){
         if(a.x[p] < b.x[p]){
            flag = false;
            break;
         }
      }
      return flag;
   }
   //================================================================================

   /*--- vecQ ---*/
   template <typename T, typename U>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline vecQ<T> operator + (vecQ<T> a, U b)
   {
     vecQ<T> c;
     for(int q = 0; q < Q; q++){
        c.f[q] = a.f[q] + b;
     }
     return c;
   }
   //================================================================================

   template <typename T, typename U>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline vecQ<T> operator - (vecQ<T> a, U b)
   {
     vecQ<T> c;
     for(int q = 0; q < Q; q++){
        c.f[q] = a.f[q] - b;
     }
     return c;
   }
   //================================================================================

   template <typename T, typename U>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline vecQ<T> operator * (vecQ<T> a, U b)
   {
     vecQ<T> c;
     for(int q = 0; q < Q; q++){
        c.f[q] = a.f[q] * b;
     }
     return c;
   }
   //================================================================================

   template <typename T, typename U>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline vecQ<T> operator / (vecQ<T> a, U b)
   {
     vecQ<T> c;
     for(int q = 0; q < Q; q++){
        c.f[q] = a.f[q] / b;
     }
     return c;
   }
   //================================================================================

   template <typename T, typename U>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline vecQ<U> vecQ_cast(vecQ<T> a)
   {
     vecQ<U> c;
     for(int q = 0; q < Q; q++){
        c.f[q] = static_cast<U>(a.f[q]);
     }
     return c;
   }
   //================================================================================

   template <typename T>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline vecQ<T> operator + (vecQ<T> a, vecQ<T> b)
   {
     vecQ<T> c;
     for(int q = 0; q < Q; q++){
        c.f[q] = a.f[q] + b.f[q];
     }
     return c;
   }
   //================================================================================

   template <typename T>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline vecQ<T> operator - (vecQ<T> a, vecQ<T> b)
   {
     vecQ<T> c;
     for(int q = 0; q < Q; q++){
        c.f[q] = a.f[q] - b.f[q];
     }
     return c;
   }
   //================================================================================

   template <typename T>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline vecQ<T> operator * (vecQ<T> a, vecQ<T> b)
   {
     vecQ<T> c;
     for(int q = 0; q < Q; q++){
        c.f[q] = a.f[q] * b.f[q];
     }
     return c;
   }
   //================================================================================

   template <typename T>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline vecQ<T> operator / (vecQ<T> a, vecQ<T> b)
   {
     vecQ<T> c;
     for(int q = 0; q < Q; q++){
        c.f[q] = a.f[q] / b.f[q];
     }
     return c;
   }
   //================================================================================

   template <typename T>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline T sum(vecQ<T> f)
   {
      T c = 0;
      for(int q = 0; q < Q; q++){
         c += f.f[q];
      }
      return c;
   }
   //================================================================================

//   template <typename T>
//#ifdef GPURUN
//   __device__ __host__
//#endif 
//   inline vecP<T> sumU(LBM<T>& lbm, vecQ<T> f)
//   {
//      vecP<T> c;
//      for(int q = 0; q < Q; q++){
//         for(int i = 0; i < P; i++){
//            c.x[i] += f.f[q]*lbm.ci[q].x[i];
//         }
//      }
//      return c;
//   }
//   //================================================================================

   /*--- vec3 ---*/
   template <typename T>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline T dot(vec3<T> a, vec3<T> b)
   {
      T c = 0;
      for(int i = 0; i < 3; i++){
         c += a.x[i]*b.x[i];
      }
      return c;
   }
   //================================================================================

   template <typename T>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline T dot(vec3<T> a, vec3<int> b)
   {
      T c = 0;
      for(int i = 0; i < 3; i++){
         c += a.x[i]*b.x[i];
      }
      return c;
   }
   //================================================================================

   template <typename T>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline vec3<T> cross(vec3<T> a, vec3<T> b)
   {
      vec3<T> c;
      c.x[0] = a.x[1]*b.x[2] - a.x[2]*b.x[1];
      c.x[1] = a.x[2]*b.x[0] - a.x[0]*b.x[2];
      c.x[2] = a.x[0]*b.x[1] - a.x[1]*b.x[0];
      return c;
   }
   //================================================================================
   
   template <typename T>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline T norm(vec3<T> a)
   {
      T c = sqrt(dot<T>(a, a));
      return c;
   }
   //================================================================================

   template <typename T>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline T norm2(vec3<T> a)
   {
      return dot<T>(a,a);
   }
   //================================================================================

   template <typename T, typename U>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline vec3<T> operator + (vec3<T> a, U b)
   {
     vec3<T> c;
     for(int i = 0; i < 3; i++){
        c.x[i] = a.x[i] + b;
     }
     return c;
   }
   //================================================================================

   template <typename T, typename U>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline vec3<T> operator - (vec3<T> a, U b)
   {
     vec3<T> c;
     for(int i = 0; i < 3; i++){
        c.x[i] = a.x[i] - b;
     }
     return c;
   }
   //================================================================================

   template <typename T, typename U>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline vec3<T> operator * (vec3<T> a, U b)
   {
     vec3<T> c;
     for(int i = 0; i < 3; i++){
        c.x[i] = a.x[i] * b;
     }
     return c;
   }
   //================================================================================

   template <typename T, typename U>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline vec3<T> operator / (vec3<T> a, U b)
   {
     vec3<T> c;
     for(int i = 0; i < 3; i++){
        c.x[i] = a.x[i] / b;
     }
     return c;
   }
   //================================================================================

   template<typename T, typename U>
#ifdef GPURUN
   __device__ __host__
#endif
   inline vec3<U> vec3_cast(vec3<T> a)
   {
     vec3<U> c;
     for(int i = 0; i < 3; i++){
        c.x[i] = static_cast<U>(a.x[i]);
     }
     return c;
   }
   //================================================================================

   template <typename T>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline vec3<T> operator + (vec3<T> a, vec3<T> b)
   {
     vec3<T> c;
     for(int i = 0; i < 3; i++){
        c.x[i] = a.x[i] + b.x[i];
     }
     return c;
   }
   //================================================================================

   template <typename T>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline vec3<T> operator - (vec3<T> a, vec3<T> b)
   {
     vec3<T> c;
     for(int i = 0; i < 3; i++){
        c.x[i] = a.x[i] - b.x[i];
     }
     return c;
   }
   //================================================================================

   template <typename T>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline vec3<T> operator * (vec3<T> a, vec3<T> b)
   {
     vec3<T> c;
     for(int i = 0; i < 3; i++){
        c.x[i] = a.x[i] * b.x[i];
     }
     return c;
   }
   //================================================================================

   template <typename T>
#ifdef GPURUN
   __device__ __host__
#endif 
   inline vec3<T> operator / (vec3<T> a, vec3<T> b)
   {
     vec3<T> c;
     for(int i = 0; i < 3; i++){
        c.x[i] = a.x[i] / b.x[i];
     }
     return c;
   }
   //================================================================================

   template <typename T>
#ifdef GPURUN
   __device__ __host__
#endif
   inline bool operator == (vec3<T> a, vec3<T> b)
   {
      bool flag = true;
      for(int i = 0; i < 3; i++){
         if(std::abs(a.x[i] - b.x[i]) > TOL){
            flag = false;
            break;
         }
      }
      return flag;
   }
   //================================================================================

   template <typename T>
#ifdef GPURUN
   __device__ __host__
#endif
   inline bool operator < (vec3<T> a, vec3<T> b)
   {
      bool flag = true;
      for(int i = 0; i < 3; i++){
         if(a.x[i] >= b.x[i]){
            flag = false;
            break;
         }
      }
      return flag;
   }
   //================================================================================

   template <typename T>
#ifdef GPURUN
   __device__ __host__
#endif
   inline bool operator > (vec3<T> a, vec3<T> b)
   {
      bool flag = true;
      for(int i = 0; i < 3; i++){
         if(a.x[i] <= b.x[i]){
            flag = false;
            break;
         }
      }
      return flag;
   }
   //================================================================================

   template <typename T>
#ifdef GPURUN
   __device__ __host__
#endif
   inline bool operator <= (vec3<T> a, vec3<T> b)
   {
      bool flag = true;
      for(int i = 0; i < 3; i++){
         if(a.x[i] > b.x[i]){
            flag = false;
            break;
         }
      }
      return flag;
   }
   //================================================================================

   template <typename T>
#ifdef GPURUN
   __device__ __host__
#endif
   inline bool operator >= (vec3<T> a, vec3<T> b)
   {
      bool flag = true;
      for(int i = 0; i < 3; i++){
         if(a.x[i] < b.x[i]){
            flag = false;
            break;
         }
      }
      return flag;
   }
   //================================================================================
}
