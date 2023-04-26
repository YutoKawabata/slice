#pragma once

#include "mydef.hpp"

namespace my_lib
{
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//
   template <typename T>
   class Output
   {
      public:
         Output(){};
         ~Output(){};
         static void write_tec(Object<T>& obj);
         static bool get_flag(std::string target, vec3<T>& x, vec3<T>& y);
         static void write_dat(Object<T>& obj, std::string target);
   };
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//
}
