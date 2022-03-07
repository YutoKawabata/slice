#include <iostream>
#include <cmath>

#include "mydef.hpp"

namespace my_lib
{
   template <typename T>
   void Slice<T>::slicer(Object<T>& obj, const vec3<T>& n, const T& d)
   {
      vec3<T>* tmp;
      bool*    tmp_flag;
      int*     tmp_ele;
      MallocHost(&tmp, obj.num_ele*3);
      MallocHost(&tmp_flag, obj.num_ele*3);
      MallocHost(&tmp_ele, obj.num_ele*2);
      for(int i = 0; i < obj.num_ele; i++){
         for(int j = 0; j < 3; j++){
            tmp_flag[i*3 + j] = false;
            vec3<T> x1 = obj.node[obj.ele[i].x[j]];
            vec3<T> x2 = obj.node[obj.ele[i].x[(j+1) % 3]] - x1;
            T e = dot(n, x1);
            T f = dot(n, x2);
            T t = 0;
            if(std::abs(f) < TOL) continue;
            t = (d - e) / f;
            if(t < 0.0 || t > 1.0) continue;
//            std::cout << "[i, j] = [" << i << "," << j
//               << "] f = " << f << ", d - e = " << d - e << ", t = " << t << "\n";
            tmp[i*3 + j] = x1 + x2*t; 
            tmp_flag[i*3 + j] = true;
//            std::cout << "x = (" << tmp[i*3 + j].x[0] << "," << tmp[i*3 + j].x[1] << "," << tmp[i*3 + j].x[2] << ")\n";
         }
      }

      obj.num_line_ele = 0;
      for(int i = 0; i < obj.num_ele; i++){
         tmp_ele[i*2 + 0] = -1;
         tmp_ele[i*2 + 1] = -1;
         if(tmp_flag[i*3 + 0] && tmp_flag[i*3 + 1] && tmp_flag[i*3 + 2]){
            if     (tmp[i*3 + 0] == tmp[i*3 + 1]) tmp_flag[i*3 + 0] = false; 
            else if(tmp[i*3 + 1] == tmp[i*3 + 2]) tmp_flag[i*3 + 1] = false;
            else if(tmp[i*3 + 2] == tmp[i*3 + 0]) tmp_flag[i*3 + 2] = false;
         }
         if(tmp_flag[i*3 + 0] && tmp_flag[i*3 + 1]){
            if(tmp[i*3 + 0] == tmp[i*3 + 1]){
               tmp_flag[i*3 + 0] = false; 
               tmp_flag[i*3 + 1] = false; 
//               std::cout << "x1 = (" << tmp[i*3 + 0].x[0] << "," << tmp[i*3 + 0].x[1] << "," << tmp[i*3 + 0].x[2] << ")\n";
//               std::cout << "x2 = (" << tmp[i*3 + 1].x[0] << "," << tmp[i*3 + 1].x[1] << "," << tmp[i*3 + 1].x[2] << ")\n";
            }else{
               tmp_ele [i*2 + 0] = i*3 + 0;
               tmp_ele [i*2 + 1] = i*3 + 1;
               obj.num_line_ele ++;
            }
         }else if(tmp_flag[i*3 + 1] && tmp_flag[i*3 + 2]){
            if(tmp[i*3 + 1] == tmp[i*3 + 2]){
               tmp_flag[i*3 + 1] = false; 
               tmp_flag[i*3 + 2] = false; 
            }else{
               obj.num_line_ele ++;
               tmp_ele [i*2 + 0] = i*3 + 1;
               tmp_ele [i*2 + 1] = i*3 + 2;
            }
         }else if(tmp_flag[i*3 + 2] && tmp_flag[i*3 + 0]){
            if(tmp[i*3 + 2] == tmp[i*3 + 0]){
               tmp_flag[i*3 + 2] = false; 
               tmp_flag[i*3 + 0] = false; 
            }else{
               obj.num_line_ele ++;
               tmp_ele [i*2 + 0] = i*3 + 2;
               tmp_ele [i*2 + 1] = i*3 + 0;
            }
         }else{
            tmp_flag[i*3 + 0] = false;
            tmp_flag[i*3 + 1] = false; 
            tmp_flag[i*3 + 2] = false; 
         }
      }
      MallocHost(&obj.line_ele,  obj.num_line_ele*2);
      MallocHost(&obj.line_node, obj.num_line_ele*2);
      for(int i = 0; i < obj.num_line_ele; i++){
         obj.line_ele[i*2 + 0] = -1;
         obj.line_ele[i*2 + 1] = -1;
      }
      obj.num_line_ele = 0;
      obj.num_line_node = 0;
      
      for(int i = 0; i < obj.num_ele; i++){
         for(int k1 = 0; k1 < 2; k1++){
            int m = tmp_ele[i*2 + k1];
            if(m < 0) continue;
            if(!tmp_flag[m]) continue;
            if(m < obj.num_line_node) continue;
            vec3<T> pos1 = tmp[m];
            obj.line_node[obj.num_line_node] = pos1;
            tmp_ele[i*2 + k1] = obj.num_line_node;
            
            for(int j = i; j < obj.num_ele; j++){
               for(int k2 = 0; k2 < 2; k2++){
                  int n = tmp_ele[j*2 + k2];
                  if(n < 0) continue;
                  if(n < obj.num_line_node) continue;
                  if(!tmp_flag[n]) continue;
                  vec3<T> pos2 = tmp[n];
                  if(pos1 == pos2){
                     tmp_ele[j*2 + k2] = obj.num_line_node;
                     break;
                  }
               }
            }
            obj.num_line_node ++;
         }
         tmp_flag[i*3 + 0] = false;
         tmp_flag[i*3 + 1] = false;
         tmp_flag[i*3 + 2] = false;
      } 

      for(int i = 0; i < obj.num_ele; i++){
         if(tmp_ele[i*2 + 1] < 0) continue;
         for(int j = i; j < obj.num_ele; j++){
            if(tmp_ele[i*2 + 1] == tmp_ele[j*2 + 0]){
               if(tmp_ele[i*2 + 0] == tmp_ele[j*2 + 1]){
                  if(tmp_flag[j*3]){
                     tmp_ele[i*2 + 0] = -1;
                     tmp_ele[i*2 + 1] = -1; 
                     break;
                  }else{
                     tmp_ele[j*2 + 0] = -1;
                     tmp_ele[j*2 + 1] = -1; 
                  }
               }else{
                  tmp_flag[j*3] = true;
               }
            }
         }
      }
            
//      for(int i = 0; i < obj.num_ele; i++){
//         std::cout << "[" << tmp_ele[i*2 + 0] << "," << tmp_ele[i*2 + 1] <<"]";
//         std::cout << "[" << tmp_flag[i*3 + 0] << "," << tmp_flag[i*3 + 1] << "," << tmp_flag[i*3 + 2] << "]\n";
//      }

      for(int i = 0; i < obj.num_ele; i++){
         if(tmp_ele[i*2 + 0] >= 0 && tmp_ele[i*2 + 1] >= 0){
            obj.line_ele[obj.num_line_ele*2 + 0] = tmp_ele[i*2 + 0];
            obj.line_ele[obj.num_line_ele*2 + 1] = tmp_ele[i*2 + 1];
            obj.num_line_ele ++;
         }
      }

      std::cout << "num_line_ele = " << obj.num_line_ele << "\n";
      std::cout << "num_line_node = " << obj.num_line_node << "\n";
   }
   //================================================================================

   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//
   template class Slice<double>;
   template class Slice<float>;
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//
}
