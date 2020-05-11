"""
1:2 binding system solved using Lagrange multiplier approach

Modified Factory example utilising Lagrane multiplier to solve complex
concentration in a 1:2 protein:ligand binding system
"""
from scipy.optimize import fsolve
from autograd import grad, jacobian
def lagrange_1_to_2(p0, l0, kd1, kd2):
    def F(X): # Augmented Lagrange function
        pf=X[0]
        lf=X[1]
        pl1=pf*lf/kd1
        pl2=pf*lf/kd2
        pl12=(pl1*lf + pl2*lf)/(kd1+kd2)
        constraint1=p0-(pf+pl1+pl2+pl12)
        constraint2=l0-(lf+pl1+pl2+2*pl12)
        return pl12 + X[2] *constraint1 + X[3]*constraint2
    dfdL = grad(F, 0) # Gradients of the Lagrange function

    pf, lf, lam1, lam2= fsolve(dfdL, [p0, l0]+[1.0]*2, fprime=jacobian(dfdL))
    pl1=pf*lf/kd1
    pl2=pf*lf/kd2
    pl12=(pl1*lf + pl2*lf)/(kd1+kd2)
    return {'pf':pf,'lf':lf, 'pl1':pl1, 'pl2':pl2,'pl12':pl12}

if __name__ == "__main__":
    concentrations=lagrange_1_to_2(14.3,8.8,5.5,3.8)
    print(f"{'Concentrations are: ':>20}{concentrations}")
    print(concentrations['pf']+concentrations['pl1']+concentrations['pl2']+concentrations['pl12'])
