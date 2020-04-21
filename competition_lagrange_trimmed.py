"""
1:1:1 competition binding system solved using Lagrange multiplier approach

Modified Factory example utilising Lagrane multiplier to solve PL complex
concentration in a 1:1: protein:ligand and protein:inhibitor binding system
"""


from scipy.optimize import fsolve
import autograd.numpy as np
from autograd import grad
def lagrange_competition(p0,l0, i0,kdpl, kdpi):
    def F(X): # Augmented Lagrange function
        response=(X[0]*X[1])/kdpl
        constraint1=p0-(response+X[0]+(X[0]*X[2]/kdpi))
        constraint2=l0-(response+X[1])
        constraint3=i0-(X[2]+(X[0]*X[2]/kdpi))
        return response - X[3] *constraint1 - X[4]*constraint2 -X[5]*constraint3
    dfdL = grad(F, 0) # Gradients of the Lagrange function
    pf, lf, inhf, lam1, lam2,lam3 = fsolve(dfdL, [p0, l0, i0, 1.0,1.0,1.0])
    return {'pf':pf,'lf':lf,'pl':(pf*lf)/kdpl}

if __name__ == "__main__":
    concentrations=lagrange_competition(10,10,10,5,1)
    print(f"The answer is at {concentrations['pl']}")