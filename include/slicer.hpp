#pragma once

#include "mydef.hpp"

namespace my_lib
{
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//
   template <typename T>
   class Slice
   {
      public:
         Slice(){};
         ~Slice(){};
         static void slicer(Object<T>& obj, const vec3<T>& n, const T& d);
   };
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//
}

