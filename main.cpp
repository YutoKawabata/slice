#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cassert>

#include "mydef.hpp"

#ifdef GPURUN
#include <cuda.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#endif

#ifdef CALC_DOUBLE
using LTYPE=double;
#else
using LTYPE=float;
#endif

int main(){
   my_lib::vec3<LTYPE> n;
   n.x[0] = 0.0;
   n.x[1] = 1.0;
   n.x[2] = 0.0;
   const LTYPE d = 0.0; //n.x[0]x + n.x[1]y + n.x[2]z = d 
   my_lib::Object<LTYPE> obj;
   obj.name = "cell0000";
   std::cout << obj.name << std::endl;
   my_lib::Input<LTYPE>::read3D_tec_tri(obj);
//   my_lib::Input<LTYPE>::read3D_ply(obj);
   my_lib::Slice<LTYPE>::slicer(obj, n, d);
   my_lib::Output<LTYPE>::write_dat(obj, "X_min"); //X_min, X_max, Y_min, Y_max, Z_min, Z_max
   my_lib::Output<LTYPE>::write_tec(obj);
   return 0;
}
