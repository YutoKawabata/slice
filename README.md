# slice

@ Slicer for 3D objects written by dat (tecplot, FEquadrilateral) or ply formats.

# Version

 ver |   Update
=====|============
 1.0 | 2022/03/07

#Development Environment

@Homebrew gcc 11.2.0_3 
@VIM 8.2
@macOS Monterey 12.0.1    

# Usage

1. 'cd "3Dfile" ./meshes'

2. Rewrite "obj.name" in main.cpp (only name, not required extension.)  

3. Choose a method from Input's member functions depending on the extension.
   ex. dat file -> Input<LTYPE>::read3D_tec_quad(obj)
       ply file -> Input<LTYPE>::read3D_ply(obj)

4. Define slicing plane by giving the normal vector "n" and a scalar "d".
   Equation for a plane: n.x[0]x + n.x[1]y + n.x[2]z = d

5. If want to calc in single precision, please comment out "#define CALC_DOUBLE" in main.hpp

6. 'Makefile'

7. './run'

#Author

Yuto Kawabata
AFE Kobe Univ.
