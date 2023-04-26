#include <iostream>
#include <istream>
#include <cmath>
#include <cassert>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include "mydef.hpp"

namespace my_lib
{
   template <typename T>
   void Swap(T& var)
   {
      char* varArray = reinterpret_cast<char*>(&var);
      for (long i = 0; i < static_cast<long>(sizeof(var)/2); i++){
         std::swap(varArray[sizeof(var) - 1 - i], varArray[i]);
      }
   }

   template <typename T>
   void Input<T>::read3D_ply(Object<T>& obj)
   {
      std::string filename = "./meshes/" + obj.name + ".ply", str;
      std::ifstream ifs;
      ifs.open(filename, std::ios::in);
      assert(!ifs.fail());
      std::vector<vec3<int> > tmp_ele;
      std::vector<vec3<T>   > tmp_node;
      while(std::getline(ifs, str)){
         std::stringstream ss{str};
         if(str == "end_header") break;
         while(std::getline(ss, str, ' ')){
            if(str == "vertex"){
               std::getline(ss, str, ' ');
//               std::cout << "vertex = " << str << std::endl;
               obj.num_node = std::stoi(str);
            }else if(str == "face"){
               std::getline(ss, str, ' ');
//               std::cout << "face = " << str << std::endl;
               obj.num_ele = std::stoi(str);
            }
         }
      }
      for(int i = 0; i < obj.num_node; i++){
         vec3<T> tmp;
         ifs >> tmp.x[0] >> tmp.x[1] >> tmp.x[2];
//         std::cout << tmp.x[0] << ":" << tmp.x[1] << ":" << tmp.x[2] << std::endl;
         std::getline(ifs, str);
         tmp_node.push_back(tmp);
      }
      for(int i = 0; i < obj.num_ele; i++){
         vec3<int> tmp;
         double n;
         ifs >> n >> tmp.x[0] >> tmp.x[1] >> tmp.x[2];
         tmp_ele.push_back(tmp);
         if(n == 4){
            tmp.x[1] = tmp.x[2]; 
            ifs >> tmp.x[2];
            tmp_ele.push_back(tmp);
 //           std::cout << tmp.x[0] << ":" << tmp.x[1] << ":" << tmp.x[2] << std::endl;
         }else if(n > 4){
            std::cout << "vertex error!! exit." << std::endl;
            exit(1);
         }
      }
      ifs.close();
      obj.num_ele = tmp_ele.size();
      MallocHost(&obj.node, obj.num_node);
      MallocHost(&obj.ele,  obj.num_ele );
      copy(tmp_node.begin(), tmp_node.end(), obj.node);
      copy(tmp_ele.begin(),  tmp_ele.end(),  obj.ele );
      std::cout << "node size = "    << obj.num_node << std::endl;
      std::cout << "element size = " << obj.num_ele  << std::endl;
   }
   //================================================================================
   
   template <typename T>
   void Input<T>::read3D_tec_quad(Object<T>& obj)
   {
      std::string filename = "./meshes/" + obj.name + ".dat", str;
      std::ifstream ifs;
      ifs.open(filename, std::ios::in);
      assert(!ifs.fail());
      std::vector<vec3<int> > tmp_ele;
      std::vector<vec3<T>   > tmp_node;
      while(std::getline(ifs, str)){
         std::stringstream ss{str};
         bool flag = false;
         while(std::getline(ss, str, ' ')){
            if(str == "NODES"){
               std::getline(ss, str, ' ');
               std::getline(ss, str, ' ');
               str.erase(str.find(','));
               obj.num_node = std::stoi(str);
            }else if (str == "FEQUADRILATERAL"){
               flag = true;
               break;
            }
         }
         if(flag) break;
      }
      for(int i = 0; i < obj.num_node; i++){
         vec3<T> tmp;
         ifs >> tmp.x[0] >> tmp.x[1] >> tmp.x[2];
//         std::cout << tmp.x[0] << ":" << tmp.x[1] << ":" << tmp.x[2] << std::endl;
         std::getline(ifs, str);
         tmp_node.push_back(tmp);
      }
      while(std::getline(ifs, str)){
         std::stringstream ss{str};
         int ele[4] = {};   
         for(int i = 0; i < 4; i++){
            std::getline(ss, str, ' ');
            ele[i] = std::stoi(str);
         }
         vec3<int> tmp;
         for(int i = 0; i < 2; i++){
            tmp.x[0] = ele[0] - 1;
            tmp.x[1] = ele[i + 1] - 1;
            tmp.x[2] = ele[i + 2] - 1;
            tmp_ele.push_back(tmp);
         }
      }
      ifs.close();
      obj.num_ele = tmp_ele.size();
      MallocHost(&obj.node, obj.num_node);
      MallocHost(&obj.ele,  obj.num_ele );
      copy(tmp_node.begin(), tmp_node.end(), obj.node);
      copy(tmp_ele.begin(),  tmp_ele.end(),  obj.ele );
      std::cout << "node size = "    << obj.num_node << std::endl;
      std::cout << "element size = " << obj.num_ele  << std::endl;
   }
   //================================================================================

   template <typename T>
   void Input<T>::read3D_vtk_sheet(Object<T>& obj)
   {
      std::string filename = "./meshes/" + obj.name + ".vtk", str;
      std::ifstream ifs;
      ifs.open(filename, std::ios::in | std::ios::binary);
      assert(!ifs.fail());
      {
         for (int i = 0; i < 4; i++) {
            getline(ifs, str);
         }
         ifs >> str >> obj.num_node >> str;
         char c;
         ifs.read(&c, sizeof(char)); 
      }
      {
         MallocHost(&obj.node, obj.num_node);
         for (int i = 0; i < obj.num_node; i++) {
            T tmpx, tmpy, tmpz;
            ifs.read(reinterpret_cast<char*>(&tmpx), sizeof(T)); 
            ifs.read(reinterpret_cast<char*>(&tmpy), sizeof(T)); 
            ifs.read(reinterpret_cast<char*>(&tmpz), sizeof(T)); 
            Swap(tmpx); Swap(tmpy); Swap(tmpz);
            obj.node[i].x[0] = tmpx;
            obj.node[i].x[1] = tmpy;
            obj.node[i].x[2] = tmpz;
         }
      }
      {
         int num_cell_data;
         char c;
         ifs >> str >> obj.num_ele >> num_cell_data;
         MallocHost(&obj.ele,  obj.num_ele*2);
         ifs.read(&c, sizeof(char)); 

         for (int i = 0; i < obj.num_ele; i++) {
            int tmp;
            int tmp_ele[4];
            ifs.read(reinterpret_cast<char*>(&tmp), sizeof(int)); 
            Swap(tmp);
            for (int j = 0; j < tmp; j++) {
               ifs.read(reinterpret_cast<char*>(&tmp_ele[j]), sizeof(int)); 
               Swap(tmp_ele[j]);
            }
            for(int j = 0; j < 2; j++){ 
               obj.ele[j + i*2].x[0] = tmp_ele[0];
               obj.ele[j + i*2].x[1] = tmp_ele[j + 1];
               obj.ele[j + i*2].x[2] = tmp_ele[j + 2];
            }
         }
         obj.num_ele *= 2;
      }
      ifs.close();
      std::cout << "node size = "    << obj.num_node << std::endl;
      std::cout << "element size = " << obj.num_ele  << std::endl;
   }
   //================================================================================
   
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//
   template class Input<double>;
   template class Input<float>;
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//
}

