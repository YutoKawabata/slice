#include <iostream>
#include <cmath>
#include <cassert>
#include <fstream>
#include <string>
#include <cstring>

#include "mydef.hpp"

namespace my_lib
{
   template <typename T>
   void Output<T>::write_tec(Object<T>& obj)
   {
      std::string filename = "./results/tec_" + obj.name + ".dat";
      std::ofstream ofs;
      ofs.open(filename, std::ios::out);
      assert(!ofs.fail());
      ofs << "TITLE = \"Slice_line\"\n";
      ofs << "VARIABLES = \"X\"\n" << "\"Y\"\n" << "\"Z\"\n";
      ofs << "ZONE T = \"Object\"\n";
      ofs << "Nodes=" << obj.num_line_node << " Elements=" << obj.num_line_ele << " ZONETYPE=FELineSeg\n";
      ofs << "DATAPACKING=POINT\n";
      for(int i = 0; i < obj.num_line_node; i++){
         for(int j = 0; j < 3; j++){
            if(std::abs(obj.line_node[i].x[j]) < TOL) obj.line_node[i].x[j] = 0.0;
         }
         ofs << obj.line_node[i].x[0] << " " << obj.line_node[i].x[1] << " " << obj.line_node[i].x[2] << "\n";  
      }
      for(int i = 0; i < obj.num_line_ele; i++){
         ofs << obj.line_ele[i*2 + 0] + 1 << " " << obj.line_ele[i*2 + 1] + 1 << "\n";  
      }
      ofs.close();
   }
   //================================================================================

   template <typename T>
   bool Output<T>::get_flag(std::string target, vec3<T>& x, vec3<T>& y)
   {
      bool flag = false;
      if (target == "X_min") {
         flag = (x.x[0] > y.x[0]);
      } else if (target == "X_max") {
         flag = (x.x[0] < y.x[0]);
      } else if (target == "Y_min") {
         flag = (x.x[1] > y.x[1]);
      } else if (target == "Y_max") {
         flag = (x.x[1] < y.x[1]);
      } else if (target == "Z_min") {
         flag = (x.x[2] > y.x[2]);
      } else if (target == "Z_max") {
         flag = (x.x[2] < y.x[2]);
      }
      return flag;
   }
   //================================================================================

   template <typename T>
   void Output<T>::write_dat(Object<T>& obj, std::string target)
   {
      std::string filename = "./results/line_" + obj.name + ".dat";
      std::ofstream ofs;
      ofs.open(filename, std::ios::out);
      assert(!ofs.fail());

      int *node_flag;
      int *tmp_ele;
      vec3<T> *tmp_node;
      MallocHost(&node_flag, obj.num_line_node);
      MallocHost(&tmp_ele, obj.num_line_ele*2);
      MallocHost(&tmp_node, obj.num_line_node);
      memcpy(tmp_ele,  obj.line_ele,  sizeof(int)*obj.num_line_ele*2);
      memcpy(tmp_node, obj.line_node, sizeof(vec3<T>)*obj.num_line_node);

      int cnt = 0;
      for (int i = 0; i < obj.num_line_node; i++) {
         node_flag[i] = -1;
      }
      for (int i = 0; i < obj.num_line_node; i++) {  
         if (node_flag[i] != -1) continue;
         for (int j = 0; j < obj.num_line_node; j++) {
            if (i == j) continue;
            if (norm(tmp_node[i] - tmp_node[j]) < TOL) {
               node_flag[i] = j;
               cnt ++;
            }
         }
      }
      for (int i = 0; i < obj.num_line_node; i++) { 
         if (node_flag[i] == -1) continue;
         for (int j = 0; j < obj.num_line_ele; j++) {
            if (tmp_ele[0 + j*2] == i) tmp_ele[0 + j*2] = node_flag[i]; 
            if (tmp_ele[1 + j*2] == i) tmp_ele[1 + j*2] = node_flag[i]; 
         }
      }

      do{
         int num = -1;
         for (int i = 0; i < obj.num_line_ele*2; i++) {
            if (tmp_ele[i] != -1) {
               if (num == -1) num = tmp_ele[i];
               if (get_flag(target, tmp_node[num], tmp_node[tmp_ele[i]])) num = tmp_ele[i];
            }
         }
        if (num == -1) break;
         
         int target_ele = -1;
         do {
            bool write_flag = false;
            for (int i = 0; i < obj.num_line_ele; i++) {
               if (target_ele == i) continue;
               if (tmp_ele[0 + i*2] == num) {
                  if (target_ele == -1) {
                     num = tmp_ele[0 + i*2];
                     ofs << tmp_node[num].x[0] << " " << tmp_node[num].x[1] << " " << tmp_node[num].x[2] << "\n";  
                  }
                  num = tmp_ele[1 + i*2];
                  tmp_ele[0 + i*2] = -1; tmp_ele[1 + i*2] = -1;
                  ofs << tmp_node[num].x[0] << " " << tmp_node[num].x[1] << " " << tmp_node[num].x[2] << "\n";  
                  target_ele = i;
                  write_flag = true;
               } else if (tmp_ele[1 + i*2] == num) {
                  if (target_ele == -1) {
                     num = tmp_ele[1 + i*2];
                     ofs << tmp_node[num].x[0] << " " << tmp_node[num].x[1] << " " << tmp_node[num].x[2] << "\n";  
                  }
                  num = tmp_ele[0 + i*2];
                  tmp_ele[0 + i*2] = -1; tmp_ele[1 + i*2] = -1;
                  ofs << tmp_node[num].x[0] << " " << tmp_node[num].x[1] << " " << tmp_node[num].x[2] << "\n";  
                  target_ele = i;
                  write_flag = true;
               }
            }
            if (!write_flag) break;
         }while(true);
      }while(true);
      ofs.close();

      free(node_flag);
      free(tmp_ele);
      free(tmp_node);
   }
   //================================================================================

   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//
   template class Output<double>;
   template class Output<float>;
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//
}
