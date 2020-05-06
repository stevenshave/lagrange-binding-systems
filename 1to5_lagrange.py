"""
1:5 binding system solved using Lagrange multiplier approach
Modified Factory example utilising Lagrane multiplier to solve complex
concentration in a 1:5 protein:ligand binding system
"""

from timeit import default_timer as timer
from scipy.optimize import fsolve
import autograd.numpy as np
from autograd import grad, jacobian
def lagrange_1_to_5(p0, l0,kd1, kd2, kd3, kd4, kd5):
	def F(X): # Augmented Lagrange function
		pf=X[0]
		lf=X[1]
		pl1=pf*lf/kd1
		pl2=pf*lf/kd2
		pl3=pf*lf/kd3
		pl4=pf*lf/kd4
		pl5=pf*lf/kd5
		pl12=(pl2*lf+pl1*lf)/(kd1+kd2)
		pl13=(pl3*lf+pl1*lf)/(kd1+kd3)
		pl14=(pl4*lf+pl1*lf)/(kd1+kd4)
		pl15=(pl5*lf+pl1*lf)/(kd1+kd5)
		pl23=(pl3*lf+pl2*lf)/(kd2+kd3)
		pl24=(pl4*lf+pl2*lf)/(kd2+kd4)
		pl25=(pl5*lf+pl2*lf)/(kd2+kd5)
		pl34=(pl4*lf+pl3*lf)/(kd3+kd4)
		pl35=(pl5*lf+pl3*lf)/(kd3+kd5)
		pl45=(pl5*lf+pl4*lf)/(kd4+kd5)
		pl123=(pl23*lf+pl13*lf+pl12*lf)/(kd1+kd2+kd3)
		pl124=(pl24*lf+pl14*lf+pl12*lf)/(kd1+kd2+kd4)
		pl125=(pl25*lf+pl15*lf+pl12*lf)/(kd1+kd2+kd5)
		pl134=(pl34*lf+pl14*lf+pl13*lf)/(kd1+kd3+kd4)
		pl135=(pl35*lf+pl15*lf+pl13*lf)/(kd1+kd3+kd5)
		pl145=(pl45*lf+pl15*lf+pl14*lf)/(kd1+kd4+kd5)
		pl234=(pl34*lf+pl24*lf+pl23*lf)/(kd2+kd3+kd4)
		pl235=(pl35*lf+pl25*lf+pl23*lf)/(kd2+kd3+kd5)
		pl245=(pl45*lf+pl25*lf+pl24*lf)/(kd2+kd4+kd5)
		pl345=(pl45*lf+pl35*lf+pl34*lf)/(kd3+kd4+kd5)
		pl1234=(pl234*lf+pl134*lf+pl124*lf+pl123*lf)/(kd1+kd2+kd3+kd4)
		pl1235=(pl235*lf+pl135*lf+pl125*lf+pl123*lf)/(kd1+kd2+kd3+kd5)
		pl1245=(pl245*lf+pl145*lf+pl125*lf+pl124*lf)/(kd1+kd2+kd4+kd5)
		pl1345=(pl345*lf+pl145*lf+pl135*lf+pl134*lf)/(kd1+kd3+kd4+kd5)
		pl2345=(pl345*lf+pl245*lf+pl235*lf+pl234*lf)/(kd2+kd3+kd4+kd5)
		pl12345=(pl2345*lf+pl1345*lf+pl1245*lf+pl1235*lf+pl1234*lf)/(kd1+kd2+kd3+kd4+kd5)
		constraint1=p0-(pf+pl1+pl2+pl3+pl4+pl5+pl12+pl13+pl14+pl15+pl23+pl24+pl25+pl34+pl35+pl45+pl123+pl124+pl125+pl134+pl135+pl145+pl234+pl235+pl245+pl345+pl1234+pl1235+pl1245+pl1345+pl2345+pl12345)
		constraint2=l0-(lf+1*(pl1+pl2+pl3+pl4+pl5)+2*(pl12+pl13+pl14+pl15+pl23+pl24+pl25+pl34+pl35+pl45)+3*(pl123+pl124+pl125+pl134+pl135+pl145+pl234+pl235+pl245+pl345)+4*(pl1234+pl1235+pl1245+pl1345+pl2345)+5*(pl12345))
		nonzero_constraint=(constraint1-abs(constraint1))-(constraint2-abs(constraint2))
		return pl12345 - X[2]*constraint1 - X[3]*constraint2 - X[4]*nonzero_constraint
	dfdL = grad(F, 0) # Gradients of the Lagrange function
	pf, lf, lam1, lam2,lam3= fsolve(dfdL, [p0, l0]+[1.0]*3, fprime=jacobian(dfdL))
	pl1=pf*lf/kd1
	pl2=pf*lf/kd2
	pl3=pf*lf/kd3
	pl4=pf*lf/kd4
	pl5=pf*lf/kd5
	pl12=(pl2*lf+pl1*lf)/(kd1+kd2)
	pl13=(pl3*lf+pl1*lf)/(kd1+kd3)
	pl14=(pl4*lf+pl1*lf)/(kd1+kd4)
	pl15=(pl5*lf+pl1*lf)/(kd1+kd5)
	pl23=(pl3*lf+pl2*lf)/(kd2+kd3)
	pl24=(pl4*lf+pl2*lf)/(kd2+kd4)
	pl25=(pl5*lf+pl2*lf)/(kd2+kd5)
	pl34=(pl4*lf+pl3*lf)/(kd3+kd4)
	pl35=(pl5*lf+pl3*lf)/(kd3+kd5)
	pl45=(pl5*lf+pl4*lf)/(kd4+kd5)
	pl123=(pl23*lf+pl13*lf+pl12*lf)/(kd1+kd2+kd3)
	pl124=(pl24*lf+pl14*lf+pl12*lf)/(kd1+kd2+kd4)
	pl125=(pl25*lf+pl15*lf+pl12*lf)/(kd1+kd2+kd5)
	pl134=(pl34*lf+pl14*lf+pl13*lf)/(kd1+kd3+kd4)
	pl135=(pl35*lf+pl15*lf+pl13*lf)/(kd1+kd3+kd5)
	pl145=(pl45*lf+pl15*lf+pl14*lf)/(kd1+kd4+kd5)
	pl234=(pl34*lf+pl24*lf+pl23*lf)/(kd2+kd3+kd4)
	pl235=(pl35*lf+pl25*lf+pl23*lf)/(kd2+kd3+kd5)
	pl245=(pl45*lf+pl25*lf+pl24*lf)/(kd2+kd4+kd5)
	pl345=(pl45*lf+pl35*lf+pl34*lf)/(kd3+kd4+kd5)
	pl1234=(pl234*lf+pl134*lf+pl124*lf+pl123*lf)/(kd1+kd2+kd3+kd4)
	pl1235=(pl235*lf+pl135*lf+pl125*lf+pl123*lf)/(kd1+kd2+kd3+kd5)
	pl1245=(pl245*lf+pl145*lf+pl125*lf+pl124*lf)/(kd1+kd2+kd4+kd5)
	pl1345=(pl345*lf+pl145*lf+pl135*lf+pl134*lf)/(kd1+kd3+kd4+kd5)
	pl2345=(pl345*lf+pl245*lf+pl235*lf+pl234*lf)/(kd2+kd3+kd4+kd5)
	pl12345=(pl2345*lf+pl1345*lf+pl1245*lf+pl1235*lf+pl1234*lf)/(kd1+kd2+kd3+kd4+kd5)
	return {'pf':pf,'lf':lf, 'pl1':pl1, 'pl2':pl2, 'pl3':pl3, 'pl4':pl4, 'pl5':pl5, 'pl12':pl12, 'pl13':pl13, 'pl14':pl14, 'pl15':pl15, 'pl23':pl23, 'pl24':pl24, 'pl25':pl25, 'pl34':pl34, 'pl35':pl35, 'pl45':pl45, 'pl123':pl123, 'pl124':pl124, 'pl125':pl125, 'pl134':pl134, 'pl135':pl135, 'pl145':pl145, 'pl234':pl234, 'pl235':pl235, 'pl245':pl245, 'pl345':pl345, 'pl1234':pl1234, 'pl1235':pl1235, 'pl1245':pl1245, 'pl1345':pl1345, 'pl2345':pl2345, 'pl12345':pl12345}