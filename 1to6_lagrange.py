"""
1:6 binding system solved using Lagrange multiplier approach
Modified Factory example utilising Lagrane multiplier to solve complex
concentration in a 1:6 protein:ligand binding system
"""

from timeit import default_timer as timer
from scipy.optimize import fsolve
import autograd.numpy as np
from autograd import grad, jacobian
def lagrange_1_to_6(p0, l0,kd1, kd2, kd3, kd4, kd5, kd6):
	def F(X): # Augmented Lagrange function
		pf=X[0]
		lf=X[1]
		pl1=pf*lf/kd1
		pl2=pf*lf/kd2
		pl3=pf*lf/kd3
		pl4=pf*lf/kd4
		pl5=pf*lf/kd5
		pl6=pf*lf/kd6
		pl12=(pl2*lf+pl1*lf)/(kd1+kd2)
		pl13=(pl3*lf+pl1*lf)/(kd1+kd3)
		pl14=(pl4*lf+pl1*lf)/(kd1+kd4)
		pl15=(pl5*lf+pl1*lf)/(kd1+kd5)
		pl16=(pl6*lf+pl1*lf)/(kd1+kd6)
		pl23=(pl3*lf+pl2*lf)/(kd2+kd3)
		pl24=(pl4*lf+pl2*lf)/(kd2+kd4)
		pl25=(pl5*lf+pl2*lf)/(kd2+kd5)
		pl26=(pl6*lf+pl2*lf)/(kd2+kd6)
		pl34=(pl4*lf+pl3*lf)/(kd3+kd4)
		pl35=(pl5*lf+pl3*lf)/(kd3+kd5)
		pl36=(pl6*lf+pl3*lf)/(kd3+kd6)
		pl45=(pl5*lf+pl4*lf)/(kd4+kd5)
		pl46=(pl6*lf+pl4*lf)/(kd4+kd6)
		pl56=(pl6*lf+pl5*lf)/(kd5+kd6)
		pl123=(pl23*lf+pl13*lf+pl12*lf)/(kd1+kd2+kd3)
		pl124=(pl24*lf+pl14*lf+pl12*lf)/(kd1+kd2+kd4)
		pl125=(pl25*lf+pl15*lf+pl12*lf)/(kd1+kd2+kd5)
		pl126=(pl26*lf+pl16*lf+pl12*lf)/(kd1+kd2+kd6)
		pl134=(pl34*lf+pl14*lf+pl13*lf)/(kd1+kd3+kd4)
		pl135=(pl35*lf+pl15*lf+pl13*lf)/(kd1+kd3+kd5)
		pl136=(pl36*lf+pl16*lf+pl13*lf)/(kd1+kd3+kd6)
		pl145=(pl45*lf+pl15*lf+pl14*lf)/(kd1+kd4+kd5)
		pl146=(pl46*lf+pl16*lf+pl14*lf)/(kd1+kd4+kd6)
		pl156=(pl56*lf+pl16*lf+pl15*lf)/(kd1+kd5+kd6)
		pl234=(pl34*lf+pl24*lf+pl23*lf)/(kd2+kd3+kd4)
		pl235=(pl35*lf+pl25*lf+pl23*lf)/(kd2+kd3+kd5)
		pl236=(pl36*lf+pl26*lf+pl23*lf)/(kd2+kd3+kd6)
		pl245=(pl45*lf+pl25*lf+pl24*lf)/(kd2+kd4+kd5)
		pl246=(pl46*lf+pl26*lf+pl24*lf)/(kd2+kd4+kd6)
		pl256=(pl56*lf+pl26*lf+pl25*lf)/(kd2+kd5+kd6)
		pl345=(pl45*lf+pl35*lf+pl34*lf)/(kd3+kd4+kd5)
		pl346=(pl46*lf+pl36*lf+pl34*lf)/(kd3+kd4+kd6)
		pl356=(pl56*lf+pl36*lf+pl35*lf)/(kd3+kd5+kd6)
		pl456=(pl56*lf+pl46*lf+pl45*lf)/(kd4+kd5+kd6)
		pl1234=(pl234*lf+pl134*lf+pl124*lf+pl123*lf)/(kd1+kd2+kd3+kd4)
		pl1235=(pl235*lf+pl135*lf+pl125*lf+pl123*lf)/(kd1+kd2+kd3+kd5)
		pl1236=(pl236*lf+pl136*lf+pl126*lf+pl123*lf)/(kd1+kd2+kd3+kd6)
		pl1245=(pl245*lf+pl145*lf+pl125*lf+pl124*lf)/(kd1+kd2+kd4+kd5)
		pl1246=(pl246*lf+pl146*lf+pl126*lf+pl124*lf)/(kd1+kd2+kd4+kd6)
		pl1256=(pl256*lf+pl156*lf+pl126*lf+pl125*lf)/(kd1+kd2+kd5+kd6)
		pl1345=(pl345*lf+pl145*lf+pl135*lf+pl134*lf)/(kd1+kd3+kd4+kd5)
		pl1346=(pl346*lf+pl146*lf+pl136*lf+pl134*lf)/(kd1+kd3+kd4+kd6)
		pl1356=(pl356*lf+pl156*lf+pl136*lf+pl135*lf)/(kd1+kd3+kd5+kd6)
		pl1456=(pl456*lf+pl156*lf+pl146*lf+pl145*lf)/(kd1+kd4+kd5+kd6)
		pl2345=(pl345*lf+pl245*lf+pl235*lf+pl234*lf)/(kd2+kd3+kd4+kd5)
		pl2346=(pl346*lf+pl246*lf+pl236*lf+pl234*lf)/(kd2+kd3+kd4+kd6)
		pl2356=(pl356*lf+pl256*lf+pl236*lf+pl235*lf)/(kd2+kd3+kd5+kd6)
		pl2456=(pl456*lf+pl256*lf+pl246*lf+pl245*lf)/(kd2+kd4+kd5+kd6)
		pl3456=(pl456*lf+pl356*lf+pl346*lf+pl345*lf)/(kd3+kd4+kd5+kd6)
		pl12345=(pl2345*lf+pl1345*lf+pl1245*lf+pl1235*lf+pl1234*lf)/(kd1+kd2+kd3+kd4+kd5)
		pl12346=(pl2346*lf+pl1346*lf+pl1246*lf+pl1236*lf+pl1234*lf)/(kd1+kd2+kd3+kd4+kd6)
		pl12356=(pl2356*lf+pl1356*lf+pl1256*lf+pl1236*lf+pl1235*lf)/(kd1+kd2+kd3+kd5+kd6)
		pl12456=(pl2456*lf+pl1456*lf+pl1256*lf+pl1246*lf+pl1245*lf)/(kd1+kd2+kd4+kd5+kd6)
		pl13456=(pl3456*lf+pl1456*lf+pl1356*lf+pl1346*lf+pl1345*lf)/(kd1+kd3+kd4+kd5+kd6)
		pl23456=(pl3456*lf+pl2456*lf+pl2356*lf+pl2346*lf+pl2345*lf)/(kd2+kd3+kd4+kd5+kd6)
		pl123456=(pl23456*lf+pl13456*lf+pl12456*lf+pl12356*lf+pl12346*lf+pl12345*lf)/(kd1+kd2+kd3+kd4+kd5+kd6)
		constraint1=p0-(pf+pl1+pl2+pl3+pl4+pl5+pl6+pl12+pl13+pl14+pl15+pl16+pl23+pl24+pl25+pl26+pl34+pl35+pl36+pl45+pl46+pl56+pl123+pl124+pl125+pl126+pl134+pl135+pl136+pl145+pl146+pl156+pl234+pl235+pl236+pl245+pl246+pl256+pl345+pl346+pl356+pl456+pl1234+pl1235+pl1236+pl1245+pl1246+pl1256+pl1345+pl1346+pl1356+pl1456+pl2345+pl2346+pl2356+pl2456+pl3456+pl12345+pl12346+pl12356+pl12456+pl13456+pl23456+pl123456)
		constraint2=l0-(lf+1*(pl1+pl2+pl3+pl4+pl5+pl6)+2*(pl12+pl13+pl14+pl15+pl16+pl23+pl24+pl25+pl26+pl34+pl35+pl36+pl45+pl46+pl56)+3*(pl123+pl124+pl125+pl126+pl134+pl135+pl136+pl145+pl146+pl156+pl234+pl235+pl236+pl245+pl246+pl256+pl345+pl346+pl356+pl456)+4*(pl1234+pl1235+pl1236+pl1245+pl1246+pl1256+pl1345+pl1346+pl1356+pl1456+pl2345+pl2346+pl2356+pl2456+pl3456)+5*(pl12345+pl12346+pl12356+pl12456+pl13456+pl23456)+6*(pl123456))
		nonzero_constraint=(constraint1-abs(constraint1))-(constraint2-abs(constraint2))
		return pl123456 + X[2]*constraint1 + X[3]*constraint2 + X[4]*nonzero_constraint
	dfdL = grad(F, 0) # Gradients of the Lagrange function
	pf, lf, lam1, lam2,lam3= fsolve(dfdL, [p0, l0]+[1.0]*3, fprime=jacobian(dfdL))
	pl1=pf*lf/kd1
	pl2=pf*lf/kd2
	pl3=pf*lf/kd3
	pl4=pf*lf/kd4
	pl5=pf*lf/kd5
	pl6=pf*lf/kd6
	pl12=(pl2*lf+pl1*lf)/(kd1+kd2)
	pl13=(pl3*lf+pl1*lf)/(kd1+kd3)
	pl14=(pl4*lf+pl1*lf)/(kd1+kd4)
	pl15=(pl5*lf+pl1*lf)/(kd1+kd5)
	pl16=(pl6*lf+pl1*lf)/(kd1+kd6)
	pl23=(pl3*lf+pl2*lf)/(kd2+kd3)
	pl24=(pl4*lf+pl2*lf)/(kd2+kd4)
	pl25=(pl5*lf+pl2*lf)/(kd2+kd5)
	pl26=(pl6*lf+pl2*lf)/(kd2+kd6)
	pl34=(pl4*lf+pl3*lf)/(kd3+kd4)
	pl35=(pl5*lf+pl3*lf)/(kd3+kd5)
	pl36=(pl6*lf+pl3*lf)/(kd3+kd6)
	pl45=(pl5*lf+pl4*lf)/(kd4+kd5)
	pl46=(pl6*lf+pl4*lf)/(kd4+kd6)
	pl56=(pl6*lf+pl5*lf)/(kd5+kd6)
	pl123=(pl23*lf+pl13*lf+pl12*lf)/(kd1+kd2+kd3)
	pl124=(pl24*lf+pl14*lf+pl12*lf)/(kd1+kd2+kd4)
	pl125=(pl25*lf+pl15*lf+pl12*lf)/(kd1+kd2+kd5)
	pl126=(pl26*lf+pl16*lf+pl12*lf)/(kd1+kd2+kd6)
	pl134=(pl34*lf+pl14*lf+pl13*lf)/(kd1+kd3+kd4)
	pl135=(pl35*lf+pl15*lf+pl13*lf)/(kd1+kd3+kd5)
	pl136=(pl36*lf+pl16*lf+pl13*lf)/(kd1+kd3+kd6)
	pl145=(pl45*lf+pl15*lf+pl14*lf)/(kd1+kd4+kd5)
	pl146=(pl46*lf+pl16*lf+pl14*lf)/(kd1+kd4+kd6)
	pl156=(pl56*lf+pl16*lf+pl15*lf)/(kd1+kd5+kd6)
	pl234=(pl34*lf+pl24*lf+pl23*lf)/(kd2+kd3+kd4)
	pl235=(pl35*lf+pl25*lf+pl23*lf)/(kd2+kd3+kd5)
	pl236=(pl36*lf+pl26*lf+pl23*lf)/(kd2+kd3+kd6)
	pl245=(pl45*lf+pl25*lf+pl24*lf)/(kd2+kd4+kd5)
	pl246=(pl46*lf+pl26*lf+pl24*lf)/(kd2+kd4+kd6)
	pl256=(pl56*lf+pl26*lf+pl25*lf)/(kd2+kd5+kd6)
	pl345=(pl45*lf+pl35*lf+pl34*lf)/(kd3+kd4+kd5)
	pl346=(pl46*lf+pl36*lf+pl34*lf)/(kd3+kd4+kd6)
	pl356=(pl56*lf+pl36*lf+pl35*lf)/(kd3+kd5+kd6)
	pl456=(pl56*lf+pl46*lf+pl45*lf)/(kd4+kd5+kd6)
	pl1234=(pl234*lf+pl134*lf+pl124*lf+pl123*lf)/(kd1+kd2+kd3+kd4)
	pl1235=(pl235*lf+pl135*lf+pl125*lf+pl123*lf)/(kd1+kd2+kd3+kd5)
	pl1236=(pl236*lf+pl136*lf+pl126*lf+pl123*lf)/(kd1+kd2+kd3+kd6)
	pl1245=(pl245*lf+pl145*lf+pl125*lf+pl124*lf)/(kd1+kd2+kd4+kd5)
	pl1246=(pl246*lf+pl146*lf+pl126*lf+pl124*lf)/(kd1+kd2+kd4+kd6)
	pl1256=(pl256*lf+pl156*lf+pl126*lf+pl125*lf)/(kd1+kd2+kd5+kd6)
	pl1345=(pl345*lf+pl145*lf+pl135*lf+pl134*lf)/(kd1+kd3+kd4+kd5)
	pl1346=(pl346*lf+pl146*lf+pl136*lf+pl134*lf)/(kd1+kd3+kd4+kd6)
	pl1356=(pl356*lf+pl156*lf+pl136*lf+pl135*lf)/(kd1+kd3+kd5+kd6)
	pl1456=(pl456*lf+pl156*lf+pl146*lf+pl145*lf)/(kd1+kd4+kd5+kd6)
	pl2345=(pl345*lf+pl245*lf+pl235*lf+pl234*lf)/(kd2+kd3+kd4+kd5)
	pl2346=(pl346*lf+pl246*lf+pl236*lf+pl234*lf)/(kd2+kd3+kd4+kd6)
	pl2356=(pl356*lf+pl256*lf+pl236*lf+pl235*lf)/(kd2+kd3+kd5+kd6)
	pl2456=(pl456*lf+pl256*lf+pl246*lf+pl245*lf)/(kd2+kd4+kd5+kd6)
	pl3456=(pl456*lf+pl356*lf+pl346*lf+pl345*lf)/(kd3+kd4+kd5+kd6)
	pl12345=(pl2345*lf+pl1345*lf+pl1245*lf+pl1235*lf+pl1234*lf)/(kd1+kd2+kd3+kd4+kd5)
	pl12346=(pl2346*lf+pl1346*lf+pl1246*lf+pl1236*lf+pl1234*lf)/(kd1+kd2+kd3+kd4+kd6)
	pl12356=(pl2356*lf+pl1356*lf+pl1256*lf+pl1236*lf+pl1235*lf)/(kd1+kd2+kd3+kd5+kd6)
	pl12456=(pl2456*lf+pl1456*lf+pl1256*lf+pl1246*lf+pl1245*lf)/(kd1+kd2+kd4+kd5+kd6)
	pl13456=(pl3456*lf+pl1456*lf+pl1356*lf+pl1346*lf+pl1345*lf)/(kd1+kd3+kd4+kd5+kd6)
	pl23456=(pl3456*lf+pl2456*lf+pl2356*lf+pl2346*lf+pl2345*lf)/(kd2+kd3+kd4+kd5+kd6)
	pl123456=(pl23456*lf+pl13456*lf+pl12456*lf+pl12356*lf+pl12346*lf+pl12345*lf)/(kd1+kd2+kd3+kd4+kd5+kd6)
	return {'pf':pf,'lf':lf, 'pl1':pl1, 'pl2':pl2, 'pl3':pl3, 'pl4':pl4, 'pl5':pl5, 'pl6':pl6, 'pl12':pl12, 'pl13':pl13, 'pl14':pl14, 'pl15':pl15, 'pl16':pl16, 'pl23':pl23, 'pl24':pl24, 'pl25':pl25, 'pl26':pl26, 'pl34':pl34, 'pl35':pl35, 'pl36':pl36, 'pl45':pl45, 'pl46':pl46, 'pl56':pl56, 'pl123':pl123, 'pl124':pl124, 'pl125':pl125, 'pl126':pl126, 'pl134':pl134, 'pl135':pl135, 'pl136':pl136, 'pl145':pl145, 'pl146':pl146, 'pl156':pl156, 'pl234':pl234, 'pl235':pl235, 'pl236':pl236, 'pl245':pl245, 'pl246':pl246, 'pl256':pl256, 'pl345':pl345, 'pl346':pl346, 'pl356':pl356, 'pl456':pl456, 'pl1234':pl1234, 'pl1235':pl1235, 'pl1236':pl1236, 'pl1245':pl1245, 'pl1246':pl1246, 'pl1256':pl1256, 'pl1345':pl1345, 'pl1346':pl1346, 'pl1356':pl1356, 'pl1456':pl1456, 'pl2345':pl2345, 'pl2346':pl2346, 'pl2356':pl2356, 'pl2456':pl2456, 'pl3456':pl3456, 'pl12345':pl12345, 'pl12346':pl12346, 'pl12356':pl12356, 'pl12456':pl12456, 'pl13456':pl13456, 'pl23456':pl23456, 'pl123456':pl123456}