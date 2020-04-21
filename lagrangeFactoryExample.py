"""
Factory revenue optimisation using Lagrange multiplier approach

Example of constrained optimisation using a steel factory, where analysts have
worked out that revenue = 200*h^(2/3)*s^(1/3), where h is hours worked, and s
is tons of steel.  Constraints are that an hours work costs 20, and a ton of
steel is 2000.  Total budget is 20000.  Find h and s to maximise revenue.

"""

import autograd.numpy as np
from autograd import grad


def objective(X):
    h, s = X
    return 200*np.power(h,(2/3))*np.power(s,(1/3))

def constraints(X):
    h, s  = X
    return 20000-(20*h+170*s)

def F(L):
    'Augmented Lagrange function'
    h,s, _lambda = L
    return objective([h,s]) - _lambda * constraints([h,s])

# Gradients of the Lagrange function
dfdL = grad(F,0)

# Find L that returns all zeros in this function.
def obj(L):
    h,s, _lambda = L
    dFdh, dFds, dFdlam = dfdL(L)
    return [dFdh, dFds, constraints([h,s])]

from scipy.optimize import fsolve
h,s, _lam = fsolve(obj, [1, 1.0, 1.0])
print(f'The answer is at {h,s,_lam}')

print(f"Revenue is ${objective([667,39])}")