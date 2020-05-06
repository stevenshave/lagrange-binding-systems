"""
homodimer formation binding system solved using Lagrange multiplier approach

Lagrange multiplier example of homodimer formation system
"""

from scipy.optimize import fsolve
import autograd.numpy as np
from autograd import grad
def lagrange_homodimerformation(p0,kdpp):
    def F(X): # Augmented Lagrange function
        response=((X[0]*X[0])/kdpp)
        constraint1=p0-(response*2+X[0])
        return response - X[1] *constraint1
    dfdL = grad(F, 0) # Gradients of the Lagrange function
    pf, lam1 = fsolve(dfdL, [p0, 1.0])
    return {'pf':pf,'pp':((pf*pf)/kdpp)}

if __name__ == "__main__":
    concentrations=lagrange_homodimerformation(14.7,3.5)
    print(f"The answer is at {concentrations['pp']}")
    print(concentrations)