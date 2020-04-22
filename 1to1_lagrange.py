"""
1:1 binding system solved using Lagrange multiplier approach

Modified Factory example utilising Lagrane multiplier to solve complex
concentration in a 1:1 protein:ligand binding system
"""

from scipy.optimize import fsolve
import autograd.numpy as np
from autograd import grad
def lagrange_one_to_one(p0,l0,kd):
    def F(X): # Augmented Lagrange function
        response=(X[0]*X[1])/kd
        return response - X[2] *(p0-(response+X[0])) - X[3]*(l0-(response+X[1]))
    dfdL = grad(F, 0) # Gradients of the Lagrange function
    pf, lf, lam1, lam2 = fsolve(dfdL, [p0, l0, 1.0,1.0])
    return {'pf':pf,'lf':lf,'pl':(pf*lf)/kd}

if __name__ == "__main__":
    concentrations=lagrange_one_to_one(14,7,10)
    print(f"The answer is at {concentrations['pl']}")