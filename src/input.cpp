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
   
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//
   template class Input<double>;
   template class Input<float>;
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//
}

