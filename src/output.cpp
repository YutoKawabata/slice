#include <iostream>
#include <cmath>
#include <cassert>
#include <fstream>
#include <string>

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
      int *node_flag = new int[obj.num_line_node]{};
      int cnt = 0;
      for (int i = 0; i < obj.num_line_node; i++) {
         node_flag[i] = -1;
      }
      for (int i = 0; i < obj.num_line_node; i++) {  
         if (node_flag[i] != -1) continue;
         for (int j = 0; j < obj.num_line_node; j++) {
            if (i == j) continue;
            if (norm(obj.line_node[i] - obj.line_node[j]) < TOL) {
               node_flag[i] = j;
               cnt ++;
            }
         }
      }
      
      for (int i = 0; i < obj.num_line_node; i++) { 
         if (node_flag[i] == -1) continue;
         for (int j = 0; j < obj.num_line_ele; j++) {
            if (obj.line_ele[0 + j*2] == i) obj.line_ele[0 + j*2] = node_flag[i]; 
            if (obj.line_ele[1 + j*2] == i) obj.line_ele[1 + j*2] = node_flag[i]; 
         }
      }

      int num = 0;
      for (int i = 1; i < obj.num_line_ele*2; i++) {
         if (get_flag(target, obj.line_node[num], obj.line_node[obj.line_ele[i]])) num = obj.line_ele[i];
      }
      
      std::string filename = "./results/line_" + obj.name + ".dat";
      std::ofstream ofs;
      ofs.open(filename, std::ios::out);
      assert(!ofs.fail());
      int target_ele = -1;
      do {
         bool write_flag = false;
         for (int i = 0; i < obj.num_line_ele; i++) {
            if (target_ele == i) continue;
            if (obj.line_ele[0 + i*2] == num) {
               if (num == -1) {
                  num = obj.line_ele[0 + i*2];
                  ofs << obj.line_node[num].x[0] << " " << obj.line_node[num].x[1] << " " << obj.line_node[num].x[2] << "\n";  
               }
               num = obj.line_ele[1 + i*2];
               ofs << obj.line_node[num].x[0] << " " << obj.line_node[num].x[1] << " " << obj.line_node[num].x[2] << "\n";  
               target_ele = i;
               write_flag = true;
            } else if (obj.line_ele[1 + i*2] == num) {
               if (num == -1) {
                  num = obj.line_ele[1 + i*2];
                  ofs << obj.line_node[num].x[0] << " " << obj.line_node[num].x[1] << " " << obj.line_node[num].x[2] << "\n";  
               }
               num = obj.line_ele[0 + i*2];
               ofs << obj.line_node[num].x[0] << " " << obj.line_node[num].x[1] << " " << obj.line_node[num].x[2] << "\n";  
               target_ele = i;
               write_flag = true;
            }
         }
         if (!write_flag) break;
      }while(true);
      ofs.close();
   }
   //================================================================================

   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//
   template class Output<double>;
   template class Output<float>;
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//
}
