"""
1:1 binding system solved using Lagrange multiplier approach

Modified Factory example utilising Lagrane multiplier to solve complex
concentration in a 1:1 protein:ligand binding system
"""

from scipy.optimize import fsolve
import autograd.numpy as np
from autograd import grad
def lagrange_1_to_2(p0, l0, kd1, kd2):
    def F(X): # Augmented Lagrange function
        #X[0] = pf
        #X[1] = lf
        #X[2] = pl1
        #X[3] = pl2
        
        pf=X[0]
        lf=X[1]
        pl1=X[2]
        pl2=X[3]
        pl12=pl1*lf/kd2 + pl2*lf/kd1
        constraint1=p0-(pf+pl1+pl2)
        constraint2=l0-(lf+pl1+pl2+2*pl12)
        return pl12 - X[4] *constraint1 - X[5]*constraint2
    dfdL = grad(F, 0) # Gradients of the Lagrange function
    pf, lf, pl1, pl2, lam1, lam2 = fsolve(dfdL, [p0, l0, 1.0,1.0,1.0,1.0])
    return {'pf':pf,'lf':lf,'pl1':pl1, 'pl2':pl2, 'pl12':pl1*lf/kd2+pl2*lf/kd1}

if __name__ == "__main__":
    concentrations=lagrange_1_to_2(14,7,0.0001,0.0001)
    print(f"The answer is at {concentrations['pl12']}")