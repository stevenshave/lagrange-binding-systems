"""
1:3 binding system solved using Lagrange multiplier approach
Modified Factory example utilising Lagrane multiplier to solve complex
concentration in a 1:3 protein:ligand binding system
"""

from scipy.optimize import fsolve
from autograd import grad, jacobian
def lagrange_1_to_3(p0, l0,kd1, kd2, kd3):
	def F(X): # Augmented Lagrange function
		pf=X[0]
		lf=X[1]
		pl1=pf*lf/kd1
		pl2=pf*lf/kd2
		pl3=pf*lf/kd3
		pl12=(pl2*lf+pl1*lf)/(kd1+kd2)
		pl13=(pl3*lf+pl1*lf)/(kd1+kd3)
		pl23=(pl3*lf+pl2*lf)/(kd2+kd3)
		pl123=(pl23*lf+pl13*lf+pl12*lf)/(kd1+kd2+kd3)
		constraint1=p0-(pf+pl1+pl2+pl3+pl12+pl13+pl23+pl123)
		constraint2=l0-(lf+1*(pl1+pl2+pl3)+2*(pl12+pl13+pl23)+3*(pl123))
		nonzero_constraint=(constraint1-abs(constraint1))-(constraint2-abs(constraint2))
		return pl123 - X[2]*constraint1 - X[3]*constraint2 - X[4]*nonzero_constraint
	dfdL = grad(F, 0) # Gradients of the Lagrange function
	pf, lf, lam1, lam2,lam3= fsolve(dfdL, [p0, l0]+[1.0]*3, fprime=jacobian(dfdL))
	pl1=pf*lf/kd1
	pl2=pf*lf/kd2
	pl3=pf*lf/kd3
	pl12=(pl2*lf+pl1*lf)/(kd1+kd2)
	pl13=(pl3*lf+pl1*lf)/(kd1+kd3)
	pl23=(pl3*lf+pl2*lf)/(kd2+kd3)
	pl123=(pl23*lf+pl13*lf+pl12*lf)/(kd1+kd2+kd3)
	return {'pf':pf,'lf':lf, 'pl1':pl1, 'pl2':pl2, 'pl3':pl3, 'pl12':pl12, 'pl13':pl13, 'pl23':pl23, 'pl123':pl123}