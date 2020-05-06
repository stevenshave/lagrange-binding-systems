# lagrange-binding-systems

_Solving binding systems with lagrange multipliers_

Experimenting with Lagrange multipliers to speed up solving of complex PyBindingCurve systems which are solved with kinetics.  Preliminary investigation shows ~10x speedup over ODE-solved systems.

Includes the most common biological binding systems, as well as two codes to explore the classic Lagrange steel factory example.

The program write_1_to_n_lagrange generates functions for solving 1:n protein:ligand binding where n is 1 to 10.


## Requirements 
- autograd >= 1.3
- future >= 0.18.2
- numpy >= 1.18.3
- scipy >= 1.4.1
- matplotlib
