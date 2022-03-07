#include <iostream>
#include <cmath>
#include <cassert>
#include <fstream>
#include <string>

#include "mydef.hpp"

namespace my_lib
{
   template <typename T>
   void Output<T>::write_line(Object<T>& obj)
   {
      std::string filename = "./results/slice_" + obj.name + ".dat";
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

   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//
   template class Output<double>;
   template class Output<float>;
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//
}
