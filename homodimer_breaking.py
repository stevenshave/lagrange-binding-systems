"""
homodimer breaking system solved using Lagrange multiplier approach

Lagrange multiplier example of homodimer breaking system
"""

from scipy.optimize import fsolve
import autograd.numpy as np
from autograd import grad
def lagrange_homodimerbreaking(p0,i0,kdpp, kdpi):
    def F(X): # Augmented Lagrange function
        pp=((X[0]*X[0])/kdpp)
        pi=((X[0]*X[1])/kdpi)
        constraint1=p0-(pp*2+pi+X[0])
        constraint2=i0-(pi+X[1])
        return pp - X[2] *constraint1 - X[3]*constraint2
    dfdL = grad(F, 0) # Gradients of the Lagrange function
    pf, i_f, lam1,lam2 = fsolve(dfdL, [p0, i0, 1.0,1.0])
    return {'pf':pf,'if':i_f,'pp':((pf*pf)/kdpp), 'pi':(pf*i_f)/kdpi}

if __name__ == "__main__":
    concentrations=lagrange_homodimerbreaking(10,12,20,10)
    print(f"The answer is at {concentrations['pp']}")
    print(concentrations)