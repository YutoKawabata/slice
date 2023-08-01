#pragma once

#include "mydef.hpp"

namespace my_lib
{
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//
   template <typename T>
   class Input
   {
      public:
         Input(){};
         ~Input(){};
         static void read3D_ply(Object<T>& obj);
         static void read3D_tec_quad(Object<T>& obj);
         static void read3D_vtk_sheet(Object<T>& obj);
         static void read3D_tec_tri(Object<T>& obj);
   };
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//
}
