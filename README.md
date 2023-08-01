# slice

@ Slicer for 3D objects written by dat (tecplot: FEquadrilateral, FEtriangle), ply and vtk formats.

# Development Environment

 ++ Homebrew gcc 11.2.0_3<br>
 ++ VIM 8.2<br>
 ++ macOS Monterey 12.0.1<br>    

# Usage

1. 'cd "3Dfile" ./meshes'

2. Rewrite "obj.name" in main.cpp (only name, not required extension.)  

3. Choose a method from Input's member functions depending on the extension.<br>
   ex. dat file -> Input<LTYPE>::read3D_tec_quad(obj)<br>
       dat file -> Input<LTYPE>::read3D_tec_tri(obj)<br>
       ply file -> Input<LTYPE>::read3D_ply(obj)<br>
       vtk file -> Input<LTYPE>::read3D_vtk_sheet(obj)<br>

4. Define slicing plane by giving the normal vector "n" and a scalar "d".
   Equation for a plane: n.x[0]x + n.x[1]y + n.x[2]z = d

5. If want to calc in single precision, please comment out "#define CALC_DOUBLE" in main.hpp

6. 'Makefile'

7. './run'

# Author

Yuto Kawabata<br>
AFE Kobe Univ.

