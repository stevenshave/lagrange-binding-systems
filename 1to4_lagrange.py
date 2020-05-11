"""
1:4 binding system solved using Lagrange multiplier approach
Modified Factory example utilising Lagrane multiplier to solve complex
concentration in a 1:4 protein:ligand binding system
"""

from timeit import default_timer as timer
from scipy.optimize import fsolve
import autograd.numpy as np
from autograd import grad, jacobian
def lagrange_1_to_4(p0, l0,kd1, kd2, kd3, kd4):
	def F(X): # Augmented Lagrange function
		pf=X[0]
		lf=X[1]
		pl1=pf*lf/kd1
		pl2=pf*lf/kd2
		pl3=pf*lf/kd3
		pl4=pf*lf/kd4
		pl12=(pl2*lf+pl1*lf)/(kd1+kd2)
		pl13=(pl3*lf+pl1*lf)/(kd1+kd3)
		pl14=(pl4*lf+pl1*lf)/(kd1+kd4)
		pl23=(pl3*lf+pl2*lf)/(kd2+kd3)
		pl24=(pl4*lf+pl2*lf)/(kd2+kd4)
		pl34=(pl4*lf+pl3*lf)/(kd3+kd4)
		pl123=(pl23*lf+pl13*lf+pl12*lf)/(kd1+kd2+kd3)
		pl124=(pl24*lf+pl14*lf+pl12*lf)/(kd1+kd2+kd4)
		pl134=(pl34*lf+pl14*lf+pl13*lf)/(kd1+kd3+kd4)
		pl234=(pl34*lf+pl24*lf+pl23*lf)/(kd2+kd3+kd4)
		pl1234=(pl234*lf+pl134*lf+pl124*lf+pl123*lf)/(kd1+kd2+kd3+kd4)
		constraint1=p0-(pf+pl1+pl2+pl3+pl4+pl12+pl13+pl14+pl23+pl24+pl34+pl123+pl124+pl134+pl234+pl1234)
		constraint2=l0-(lf+1*(pl1+pl2+pl3+pl4)+2*(pl12+pl13+pl14+pl23+pl24+pl34)+3*(pl123+pl124+pl134+pl234)+4*(pl1234))
		nonzero_constraint=(constraint1-abs(constraint1))-(constraint2-abs(constraint2))
		return pl1234 + X[2]*constraint1 + X[3]*constraint2 + X[4]*nonzero_constraint
	dfdL = grad(F, 0) # Gradients of the Lagrange function
	pf, lf, lam1, lam2,lam3= fsolve(dfdL, [p0, l0]+[1.0]*3, fprime=jacobian(dfdL))
	pl1=pf*lf/kd1
	pl2=pf*lf/kd2
	pl3=pf*lf/kd3
	pl4=pf*lf/kd4
	pl12=(pl2*lf+pl1*lf)/(kd1+kd2)
	pl13=(pl3*lf+pl1*lf)/(kd1+kd3)
	pl14=(pl4*lf+pl1*lf)/(kd1+kd4)
	pl23=(pl3*lf+pl2*lf)/(kd2+kd3)
	pl24=(pl4*lf+pl2*lf)/(kd2+kd4)
	pl34=(pl4*lf+pl3*lf)/(kd3+kd4)
	pl123=(pl23*lf+pl13*lf+pl12*lf)/(kd1+kd2+kd3)
	pl124=(pl24*lf+pl14*lf+pl12*lf)/(kd1+kd2+kd4)
	pl134=(pl34*lf+pl14*lf+pl13*lf)/(kd1+kd3+kd4)
	pl234=(pl34*lf+pl24*lf+pl23*lf)/(kd2+kd3+kd4)
	pl1234=(pl234*lf+pl134*lf+pl124*lf+pl123*lf)/(kd1+kd2+kd3+kd4)
	return {'pf':pf,'lf':lf, 'pl1':pl1, 'pl2':pl2, 'pl3':pl3, 'pl4':pl4, 'pl12':pl12, 'pl13':pl13, 'pl14':pl14, 'pl23':pl23, 'pl24':pl24, 'pl34':pl34, 'pl123':pl123, 'pl124':pl124, 'pl134':pl134, 'pl234':pl234, 'pl1234':pl1234}