"""
1:1 binding system solved using Lagrange multiplier approach

Modified Factory example utilising Lagrane multiplier to solve complex
concentration in a 1:1 protein:ligand binding system.  Verbose for
understanding.
"""

from scipy.optimize import fsolve
import autograd.numpy as np
from autograd import grad

KD = 10.0
P0 = 14.0
L0 = 7.0

# X is pf, lf, pl

def objective(X):
    pf,lf=X
    # Returning conc of PL
    return (pf*lf)/KD


def c1(X):
    return (P0-(objective(X)+X[0]))
def c2(X):
    return (L0-(objective(X)+X[1]))

def F(X):
    'Augmented Lagrange function'
    pf,lf,lam1,lam2 = X    
    return objective([pf,lf]) - lam1 * c1([pf,lf]) - lam2*c2([pf,lf])


# Gradients of the Lagrange function
dfdL = grad(F, 0)

# Find L that returns all zeros in this function.
def minimiser_objective(L):
    pf, lf, lam1, lam2 = L
    dFdpf, dFdlf, dFdlam1, dFdlam2 = dfdL(L)
    return [dFdpf, dFdlf, c1([pf,lf]),c2([pf,lf])]

pf, lf, lam1, lam2 = fsolve(minimiser_objective, [P0, L0, 1.0,1.0])
print(f'The answer is at {pf,lf, objective([pf,lf])}')

