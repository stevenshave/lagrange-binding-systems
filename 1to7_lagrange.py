"""
1:7 binding system solved using Lagrange multiplier approach
Modified Factory example utilising Lagrane multiplier to solve complex
concentration in a 1:7 protein:ligand binding system
"""

from timeit import default_timer as timer
from scipy.optimize import fsolve
import autograd.numpy as np
from autograd import grad, jacobian
def lagrange_1_to_7(p0, l0,kd1, kd2, kd3, kd4, kd5, kd6, kd7):
	def F(X): # Augmented Lagrange function
		pf=X[0]
		lf=X[1]
		pl1=pf*lf/kd1
		pl2=pf*lf/kd2
		pl3=pf*lf/kd3
		pl4=pf*lf/kd4
		pl5=pf*lf/kd5
		pl6=pf*lf/kd6
		pl7=pf*lf/kd7
		pl12=(pl2*lf+pl1*lf)/(kd1+kd2)
		pl13=(pl3*lf+pl1*lf)/(kd1+kd3)
		pl14=(pl4*lf+pl1*lf)/(kd1+kd4)
		pl15=(pl5*lf+pl1*lf)/(kd1+kd5)
		pl16=(pl6*lf+pl1*lf)/(kd1+kd6)
		pl17=(pl7*lf+pl1*lf)/(kd1+kd7)
		pl23=(pl3*lf+pl2*lf)/(kd2+kd3)
		pl24=(pl4*lf+pl2*lf)/(kd2+kd4)
		pl25=(pl5*lf+pl2*lf)/(kd2+kd5)
		pl26=(pl6*lf+pl2*lf)/(kd2+kd6)
		pl27=(pl7*lf+pl2*lf)/(kd2+kd7)
		pl34=(pl4*lf+pl3*lf)/(kd3+kd4)
		pl35=(pl5*lf+pl3*lf)/(kd3+kd5)
		pl36=(pl6*lf+pl3*lf)/(kd3+kd6)
		pl37=(pl7*lf+pl3*lf)/(kd3+kd7)
		pl45=(pl5*lf+pl4*lf)/(kd4+kd5)
		pl46=(pl6*lf+pl4*lf)/(kd4+kd6)
		pl47=(pl7*lf+pl4*lf)/(kd4+kd7)
		pl56=(pl6*lf+pl5*lf)/(kd5+kd6)
		pl57=(pl7*lf+pl5*lf)/(kd5+kd7)
		pl67=(pl7*lf+pl6*lf)/(kd6+kd7)
		pl123=(pl23*lf+pl13*lf+pl12*lf)/(kd1+kd2+kd3)
		pl124=(pl24*lf+pl14*lf+pl12*lf)/(kd1+kd2+kd4)
		pl125=(pl25*lf+pl15*lf+pl12*lf)/(kd1+kd2+kd5)
		pl126=(pl26*lf+pl16*lf+pl12*lf)/(kd1+kd2+kd6)
		pl127=(pl27*lf+pl17*lf+pl12*lf)/(kd1+kd2+kd7)
		pl134=(pl34*lf+pl14*lf+pl13*lf)/(kd1+kd3+kd4)
		pl135=(pl35*lf+pl15*lf+pl13*lf)/(kd1+kd3+kd5)
		pl136=(pl36*lf+pl16*lf+pl13*lf)/(kd1+kd3+kd6)
		pl137=(pl37*lf+pl17*lf+pl13*lf)/(kd1+kd3+kd7)
		pl145=(pl45*lf+pl15*lf+pl14*lf)/(kd1+kd4+kd5)
		pl146=(pl46*lf+pl16*lf+pl14*lf)/(kd1+kd4+kd6)
		pl147=(pl47*lf+pl17*lf+pl14*lf)/(kd1+kd4+kd7)
		pl156=(pl56*lf+pl16*lf+pl15*lf)/(kd1+kd5+kd6)
		pl157=(pl57*lf+pl17*lf+pl15*lf)/(kd1+kd5+kd7)
		pl167=(pl67*lf+pl17*lf+pl16*lf)/(kd1+kd6+kd7)
		pl234=(pl34*lf+pl24*lf+pl23*lf)/(kd2+kd3+kd4)
		pl235=(pl35*lf+pl25*lf+pl23*lf)/(kd2+kd3+kd5)
		pl236=(pl36*lf+pl26*lf+pl23*lf)/(kd2+kd3+kd6)
		pl237=(pl37*lf+pl27*lf+pl23*lf)/(kd2+kd3+kd7)
		pl245=(pl45*lf+pl25*lf+pl24*lf)/(kd2+kd4+kd5)
		pl246=(pl46*lf+pl26*lf+pl24*lf)/(kd2+kd4+kd6)
		pl247=(pl47*lf+pl27*lf+pl24*lf)/(kd2+kd4+kd7)
		pl256=(pl56*lf+pl26*lf+pl25*lf)/(kd2+kd5+kd6)
		pl257=(pl57*lf+pl27*lf+pl25*lf)/(kd2+kd5+kd7)
		pl267=(pl67*lf+pl27*lf+pl26*lf)/(kd2+kd6+kd7)
		pl345=(pl45*lf+pl35*lf+pl34*lf)/(kd3+kd4+kd5)
		pl346=(pl46*lf+pl36*lf+pl34*lf)/(kd3+kd4+kd6)
		pl347=(pl47*lf+pl37*lf+pl34*lf)/(kd3+kd4+kd7)
		pl356=(pl56*lf+pl36*lf+pl35*lf)/(kd3+kd5+kd6)
		pl357=(pl57*lf+pl37*lf+pl35*lf)/(kd3+kd5+kd7)
		pl367=(pl67*lf+pl37*lf+pl36*lf)/(kd3+kd6+kd7)
		pl456=(pl56*lf+pl46*lf+pl45*lf)/(kd4+kd5+kd6)
		pl457=(pl57*lf+pl47*lf+pl45*lf)/(kd4+kd5+kd7)
		pl467=(pl67*lf+pl47*lf+pl46*lf)/(kd4+kd6+kd7)
		pl567=(pl67*lf+pl57*lf+pl56*lf)/(kd5+kd6+kd7)
		pl1234=(pl234*lf+pl134*lf+pl124*lf+pl123*lf)/(kd1+kd2+kd3+kd4)
		pl1235=(pl235*lf+pl135*lf+pl125*lf+pl123*lf)/(kd1+kd2+kd3+kd5)
		pl1236=(pl236*lf+pl136*lf+pl126*lf+pl123*lf)/(kd1+kd2+kd3+kd6)
		pl1237=(pl237*lf+pl137*lf+pl127*lf+pl123*lf)/(kd1+kd2+kd3+kd7)
		pl1245=(pl245*lf+pl145*lf+pl125*lf+pl124*lf)/(kd1+kd2+kd4+kd5)
		pl1246=(pl246*lf+pl146*lf+pl126*lf+pl124*lf)/(kd1+kd2+kd4+kd6)
		pl1247=(pl247*lf+pl147*lf+pl127*lf+pl124*lf)/(kd1+kd2+kd4+kd7)
		pl1256=(pl256*lf+pl156*lf+pl126*lf+pl125*lf)/(kd1+kd2+kd5+kd6)
		pl1257=(pl257*lf+pl157*lf+pl127*lf+pl125*lf)/(kd1+kd2+kd5+kd7)
		pl1267=(pl267*lf+pl167*lf+pl127*lf+pl126*lf)/(kd1+kd2+kd6+kd7)
		pl1345=(pl345*lf+pl145*lf+pl135*lf+pl134*lf)/(kd1+kd3+kd4+kd5)
		pl1346=(pl346*lf+pl146*lf+pl136*lf+pl134*lf)/(kd1+kd3+kd4+kd6)
		pl1347=(pl347*lf+pl147*lf+pl137*lf+pl134*lf)/(kd1+kd3+kd4+kd7)
		pl1356=(pl356*lf+pl156*lf+pl136*lf+pl135*lf)/(kd1+kd3+kd5+kd6)
		pl1357=(pl357*lf+pl157*lf+pl137*lf+pl135*lf)/(kd1+kd3+kd5+kd7)
		pl1367=(pl367*lf+pl167*lf+pl137*lf+pl136*lf)/(kd1+kd3+kd6+kd7)
		pl1456=(pl456*lf+pl156*lf+pl146*lf+pl145*lf)/(kd1+kd4+kd5+kd6)
		pl1457=(pl457*lf+pl157*lf+pl147*lf+pl145*lf)/(kd1+kd4+kd5+kd7)
		pl1467=(pl467*lf+pl167*lf+pl147*lf+pl146*lf)/(kd1+kd4+kd6+kd7)
		pl1567=(pl567*lf+pl167*lf+pl157*lf+pl156*lf)/(kd1+kd5+kd6+kd7)
		pl2345=(pl345*lf+pl245*lf+pl235*lf+pl234*lf)/(kd2+kd3+kd4+kd5)
		pl2346=(pl346*lf+pl246*lf+pl236*lf+pl234*lf)/(kd2+kd3+kd4+kd6)
		pl2347=(pl347*lf+pl247*lf+pl237*lf+pl234*lf)/(kd2+kd3+kd4+kd7)
		pl2356=(pl356*lf+pl256*lf+pl236*lf+pl235*lf)/(kd2+kd3+kd5+kd6)
		pl2357=(pl357*lf+pl257*lf+pl237*lf+pl235*lf)/(kd2+kd3+kd5+kd7)
		pl2367=(pl367*lf+pl267*lf+pl237*lf+pl236*lf)/(kd2+kd3+kd6+kd7)
		pl2456=(pl456*lf+pl256*lf+pl246*lf+pl245*lf)/(kd2+kd4+kd5+kd6)
		pl2457=(pl457*lf+pl257*lf+pl247*lf+pl245*lf)/(kd2+kd4+kd5+kd7)
		pl2467=(pl467*lf+pl267*lf+pl247*lf+pl246*lf)/(kd2+kd4+kd6+kd7)
		pl2567=(pl567*lf+pl267*lf+pl257*lf+pl256*lf)/(kd2+kd5+kd6+kd7)
		pl3456=(pl456*lf+pl356*lf+pl346*lf+pl345*lf)/(kd3+kd4+kd5+kd6)
		pl3457=(pl457*lf+pl357*lf+pl347*lf+pl345*lf)/(kd3+kd4+kd5+kd7)
		pl3467=(pl467*lf+pl367*lf+pl347*lf+pl346*lf)/(kd3+kd4+kd6+kd7)
		pl3567=(pl567*lf+pl367*lf+pl357*lf+pl356*lf)/(kd3+kd5+kd6+kd7)
		pl4567=(pl567*lf+pl467*lf+pl457*lf+pl456*lf)/(kd4+kd5+kd6+kd7)
		pl12345=(pl2345*lf+pl1345*lf+pl1245*lf+pl1235*lf+pl1234*lf)/(kd1+kd2+kd3+kd4+kd5)
		pl12346=(pl2346*lf+pl1346*lf+pl1246*lf+pl1236*lf+pl1234*lf)/(kd1+kd2+kd3+kd4+kd6)
		pl12347=(pl2347*lf+pl1347*lf+pl1247*lf+pl1237*lf+pl1234*lf)/(kd1+kd2+kd3+kd4+kd7)
		pl12356=(pl2356*lf+pl1356*lf+pl1256*lf+pl1236*lf+pl1235*lf)/(kd1+kd2+kd3+kd5+kd6)
		pl12357=(pl2357*lf+pl1357*lf+pl1257*lf+pl1237*lf+pl1235*lf)/(kd1+kd2+kd3+kd5+kd7)
		pl12367=(pl2367*lf+pl1367*lf+pl1267*lf+pl1237*lf+pl1236*lf)/(kd1+kd2+kd3+kd6+kd7)
		pl12456=(pl2456*lf+pl1456*lf+pl1256*lf+pl1246*lf+pl1245*lf)/(kd1+kd2+kd4+kd5+kd6)
		pl12457=(pl2457*lf+pl1457*lf+pl1257*lf+pl1247*lf+pl1245*lf)/(kd1+kd2+kd4+kd5+kd7)
		pl12467=(pl2467*lf+pl1467*lf+pl1267*lf+pl1247*lf+pl1246*lf)/(kd1+kd2+kd4+kd6+kd7)
		pl12567=(pl2567*lf+pl1567*lf+pl1267*lf+pl1257*lf+pl1256*lf)/(kd1+kd2+kd5+kd6+kd7)
		pl13456=(pl3456*lf+pl1456*lf+pl1356*lf+pl1346*lf+pl1345*lf)/(kd1+kd3+kd4+kd5+kd6)
		pl13457=(pl3457*lf+pl1457*lf+pl1357*lf+pl1347*lf+pl1345*lf)/(kd1+kd3+kd4+kd5+kd7)
		pl13467=(pl3467*lf+pl1467*lf+pl1367*lf+pl1347*lf+pl1346*lf)/(kd1+kd3+kd4+kd6+kd7)
		pl13567=(pl3567*lf+pl1567*lf+pl1367*lf+pl1357*lf+pl1356*lf)/(kd1+kd3+kd5+kd6+kd7)
		pl14567=(pl4567*lf+pl1567*lf+pl1467*lf+pl1457*lf+pl1456*lf)/(kd1+kd4+kd5+kd6+kd7)
		pl23456=(pl3456*lf+pl2456*lf+pl2356*lf+pl2346*lf+pl2345*lf)/(kd2+kd3+kd4+kd5+kd6)
		pl23457=(pl3457*lf+pl2457*lf+pl2357*lf+pl2347*lf+pl2345*lf)/(kd2+kd3+kd4+kd5+kd7)
		pl23467=(pl3467*lf+pl2467*lf+pl2367*lf+pl2347*lf+pl2346*lf)/(kd2+kd3+kd4+kd6+kd7)
		pl23567=(pl3567*lf+pl2567*lf+pl2367*lf+pl2357*lf+pl2356*lf)/(kd2+kd3+kd5+kd6+kd7)
		pl24567=(pl4567*lf+pl2567*lf+pl2467*lf+pl2457*lf+pl2456*lf)/(kd2+kd4+kd5+kd6+kd7)
		pl34567=(pl4567*lf+pl3567*lf+pl3467*lf+pl3457*lf+pl3456*lf)/(kd3+kd4+kd5+kd6+kd7)
		pl123456=(pl23456*lf+pl13456*lf+pl12456*lf+pl12356*lf+pl12346*lf+pl12345*lf)/(kd1+kd2+kd3+kd4+kd5+kd6)
		pl123457=(pl23457*lf+pl13457*lf+pl12457*lf+pl12357*lf+pl12347*lf+pl12345*lf)/(kd1+kd2+kd3+kd4+kd5+kd7)
		pl123467=(pl23467*lf+pl13467*lf+pl12467*lf+pl12367*lf+pl12347*lf+pl12346*lf)/(kd1+kd2+kd3+kd4+kd6+kd7)
		pl123567=(pl23567*lf+pl13567*lf+pl12567*lf+pl12367*lf+pl12357*lf+pl12356*lf)/(kd1+kd2+kd3+kd5+kd6+kd7)
		pl124567=(pl24567*lf+pl14567*lf+pl12567*lf+pl12467*lf+pl12457*lf+pl12456*lf)/(kd1+kd2+kd4+kd5+kd6+kd7)
		pl134567=(pl34567*lf+pl14567*lf+pl13567*lf+pl13467*lf+pl13457*lf+pl13456*lf)/(kd1+kd3+kd4+kd5+kd6+kd7)
		pl234567=(pl34567*lf+pl24567*lf+pl23567*lf+pl23467*lf+pl23457*lf+pl23456*lf)/(kd2+kd3+kd4+kd5+kd6+kd7)
		pl1234567=(pl234567*lf+pl134567*lf+pl124567*lf+pl123567*lf+pl123467*lf+pl123457*lf+pl123456*lf)/(kd1+kd2+kd3+kd4+kd5+kd6+kd7)
		constraint1=p0-(pf+pl1+pl2+pl3+pl4+pl5+pl6+pl7+pl12+pl13+pl14+pl15+pl16+pl17+pl23+pl24+pl25+pl26+pl27+pl34+pl35+pl36+pl37+pl45+pl46+pl47+pl56+pl57+pl67+pl123+pl124+pl125+pl126+pl127+pl134+pl135+pl136+pl137+pl145+pl146+pl147+pl156+pl157+pl167+pl234+pl235+pl236+pl237+pl245+pl246+pl247+pl256+pl257+pl267+pl345+pl346+pl347+pl356+pl357+pl367+pl456+pl457+pl467+pl567+pl1234+pl1235+pl1236+pl1237+pl1245+pl1246+pl1247+pl1256+pl1257+pl1267+pl1345+pl1346+pl1347+pl1356+pl1357+pl1367+pl1456+pl1457+pl1467+pl1567+pl2345+pl2346+pl2347+pl2356+pl2357+pl2367+pl2456+pl2457+pl2467+pl2567+pl3456+pl3457+pl3467+pl3567+pl4567+pl12345+pl12346+pl12347+pl12356+pl12357+pl12367+pl12456+pl12457+pl12467+pl12567+pl13456+pl13457+pl13467+pl13567+pl14567+pl23456+pl23457+pl23467+pl23567+pl24567+pl34567+pl123456+pl123457+pl123467+pl123567+pl124567+pl134567+pl234567+pl1234567)
		constraint2=l0-(lf+1*(pl1+pl2+pl3+pl4+pl5+pl6+pl7)+2*(pl12+pl13+pl14+pl15+pl16+pl17+pl23+pl24+pl25+pl26+pl27+pl34+pl35+pl36+pl37+pl45+pl46+pl47+pl56+pl57+pl67)+3*(pl123+pl124+pl125+pl126+pl127+pl134+pl135+pl136+pl137+pl145+pl146+pl147+pl156+pl157+pl167+pl234+pl235+pl236+pl237+pl245+pl246+pl247+pl256+pl257+pl267+pl345+pl346+pl347+pl356+pl357+pl367+pl456+pl457+pl467+pl567)+4*(pl1234+pl1235+pl1236+pl1237+pl1245+pl1246+pl1247+pl1256+pl1257+pl1267+pl1345+pl1346+pl1347+pl1356+pl1357+pl1367+pl1456+pl1457+pl1467+pl1567+pl2345+pl2346+pl2347+pl2356+pl2357+pl2367+pl2456+pl2457+pl2467+pl2567+pl3456+pl3457+pl3467+pl3567+pl4567)+5*(pl12345+pl12346+pl12347+pl12356+pl12357+pl12367+pl12456+pl12457+pl12467+pl12567+pl13456+pl13457+pl13467+pl13567+pl14567+pl23456+pl23457+pl23467+pl23567+pl24567+pl34567)+6*(pl123456+pl123457+pl123467+pl123567+pl124567+pl134567+pl234567)+7*(pl1234567))
		nonzero_constraint=(constraint1-abs(constraint1))-(constraint2-abs(constraint2))
		return pl1234567 + X[2]*constraint1 + X[3]*constraint2 + X[4]*nonzero_constraint
	dfdL = grad(F, 0) # Gradients of the Lagrange function
	pf, lf, lam1, lam2,lam3= fsolve(dfdL, [p0, l0]+[1.0]*3, fprime=jacobian(dfdL))
	pl1=pf*lf/kd1
	pl2=pf*lf/kd2
	pl3=pf*lf/kd3
	pl4=pf*lf/kd4
	pl5=pf*lf/kd5
	pl6=pf*lf/kd6
	pl7=pf*lf/kd7
	pl12=(pl2*lf+pl1*lf)/(kd1+kd2)
	pl13=(pl3*lf+pl1*lf)/(kd1+kd3)
	pl14=(pl4*lf+pl1*lf)/(kd1+kd4)
	pl15=(pl5*lf+pl1*lf)/(kd1+kd5)
	pl16=(pl6*lf+pl1*lf)/(kd1+kd6)
	pl17=(pl7*lf+pl1*lf)/(kd1+kd7)
	pl23=(pl3*lf+pl2*lf)/(kd2+kd3)
	pl24=(pl4*lf+pl2*lf)/(kd2+kd4)
	pl25=(pl5*lf+pl2*lf)/(kd2+kd5)
	pl26=(pl6*lf+pl2*lf)/(kd2+kd6)
	pl27=(pl7*lf+pl2*lf)/(kd2+kd7)
	pl34=(pl4*lf+pl3*lf)/(kd3+kd4)
	pl35=(pl5*lf+pl3*lf)/(kd3+kd5)
	pl36=(pl6*lf+pl3*lf)/(kd3+kd6)
	pl37=(pl7*lf+pl3*lf)/(kd3+kd7)
	pl45=(pl5*lf+pl4*lf)/(kd4+kd5)
	pl46=(pl6*lf+pl4*lf)/(kd4+kd6)
	pl47=(pl7*lf+pl4*lf)/(kd4+kd7)
	pl56=(pl6*lf+pl5*lf)/(kd5+kd6)
	pl57=(pl7*lf+pl5*lf)/(kd5+kd7)
	pl67=(pl7*lf+pl6*lf)/(kd6+kd7)
	pl123=(pl23*lf+pl13*lf+pl12*lf)/(kd1+kd2+kd3)
	pl124=(pl24*lf+pl14*lf+pl12*lf)/(kd1+kd2+kd4)
	pl125=(pl25*lf+pl15*lf+pl12*lf)/(kd1+kd2+kd5)
	pl126=(pl26*lf+pl16*lf+pl12*lf)/(kd1+kd2+kd6)
	pl127=(pl27*lf+pl17*lf+pl12*lf)/(kd1+kd2+kd7)
	pl134=(pl34*lf+pl14*lf+pl13*lf)/(kd1+kd3+kd4)
	pl135=(pl35*lf+pl15*lf+pl13*lf)/(kd1+kd3+kd5)
	pl136=(pl36*lf+pl16*lf+pl13*lf)/(kd1+kd3+kd6)
	pl137=(pl37*lf+pl17*lf+pl13*lf)/(kd1+kd3+kd7)
	pl145=(pl45*lf+pl15*lf+pl14*lf)/(kd1+kd4+kd5)
	pl146=(pl46*lf+pl16*lf+pl14*lf)/(kd1+kd4+kd6)
	pl147=(pl47*lf+pl17*lf+pl14*lf)/(kd1+kd4+kd7)
	pl156=(pl56*lf+pl16*lf+pl15*lf)/(kd1+kd5+kd6)
	pl157=(pl57*lf+pl17*lf+pl15*lf)/(kd1+kd5+kd7)
	pl167=(pl67*lf+pl17*lf+pl16*lf)/(kd1+kd6+kd7)
	pl234=(pl34*lf+pl24*lf+pl23*lf)/(kd2+kd3+kd4)
	pl235=(pl35*lf+pl25*lf+pl23*lf)/(kd2+kd3+kd5)
	pl236=(pl36*lf+pl26*lf+pl23*lf)/(kd2+kd3+kd6)
	pl237=(pl37*lf+pl27*lf+pl23*lf)/(kd2+kd3+kd7)
	pl245=(pl45*lf+pl25*lf+pl24*lf)/(kd2+kd4+kd5)
	pl246=(pl46*lf+pl26*lf+pl24*lf)/(kd2+kd4+kd6)
	pl247=(pl47*lf+pl27*lf+pl24*lf)/(kd2+kd4+kd7)
	pl256=(pl56*lf+pl26*lf+pl25*lf)/(kd2+kd5+kd6)
	pl257=(pl57*lf+pl27*lf+pl25*lf)/(kd2+kd5+kd7)
	pl267=(pl67*lf+pl27*lf+pl26*lf)/(kd2+kd6+kd7)
	pl345=(pl45*lf+pl35*lf+pl34*lf)/(kd3+kd4+kd5)
	pl346=(pl46*lf+pl36*lf+pl34*lf)/(kd3+kd4+kd6)
	pl347=(pl47*lf+pl37*lf+pl34*lf)/(kd3+kd4+kd7)
	pl356=(pl56*lf+pl36*lf+pl35*lf)/(kd3+kd5+kd6)
	pl357=(pl57*lf+pl37*lf+pl35*lf)/(kd3+kd5+kd7)
	pl367=(pl67*lf+pl37*lf+pl36*lf)/(kd3+kd6+kd7)
	pl456=(pl56*lf+pl46*lf+pl45*lf)/(kd4+kd5+kd6)
	pl457=(pl57*lf+pl47*lf+pl45*lf)/(kd4+kd5+kd7)
	pl467=(pl67*lf+pl47*lf+pl46*lf)/(kd4+kd6+kd7)
	pl567=(pl67*lf+pl57*lf+pl56*lf)/(kd5+kd6+kd7)
	pl1234=(pl234*lf+pl134*lf+pl124*lf+pl123*lf)/(kd1+kd2+kd3+kd4)
	pl1235=(pl235*lf+pl135*lf+pl125*lf+pl123*lf)/(kd1+kd2+kd3+kd5)
	pl1236=(pl236*lf+pl136*lf+pl126*lf+pl123*lf)/(kd1+kd2+kd3+kd6)
	pl1237=(pl237*lf+pl137*lf+pl127*lf+pl123*lf)/(kd1+kd2+kd3+kd7)
	pl1245=(pl245*lf+pl145*lf+pl125*lf+pl124*lf)/(kd1+kd2+kd4+kd5)
	pl1246=(pl246*lf+pl146*lf+pl126*lf+pl124*lf)/(kd1+kd2+kd4+kd6)
	pl1247=(pl247*lf+pl147*lf+pl127*lf+pl124*lf)/(kd1+kd2+kd4+kd7)
	pl1256=(pl256*lf+pl156*lf+pl126*lf+pl125*lf)/(kd1+kd2+kd5+kd6)
	pl1257=(pl257*lf+pl157*lf+pl127*lf+pl125*lf)/(kd1+kd2+kd5+kd7)
	pl1267=(pl267*lf+pl167*lf+pl127*lf+pl126*lf)/(kd1+kd2+kd6+kd7)
	pl1345=(pl345*lf+pl145*lf+pl135*lf+pl134*lf)/(kd1+kd3+kd4+kd5)
	pl1346=(pl346*lf+pl146*lf+pl136*lf+pl134*lf)/(kd1+kd3+kd4+kd6)
	pl1347=(pl347*lf+pl147*lf+pl137*lf+pl134*lf)/(kd1+kd3+kd4+kd7)
	pl1356=(pl356*lf+pl156*lf+pl136*lf+pl135*lf)/(kd1+kd3+kd5+kd6)
	pl1357=(pl357*lf+pl157*lf+pl137*lf+pl135*lf)/(kd1+kd3+kd5+kd7)
	pl1367=(pl367*lf+pl167*lf+pl137*lf+pl136*lf)/(kd1+kd3+kd6+kd7)
	pl1456=(pl456*lf+pl156*lf+pl146*lf+pl145*lf)/(kd1+kd4+kd5+kd6)
	pl1457=(pl457*lf+pl157*lf+pl147*lf+pl145*lf)/(kd1+kd4+kd5+kd7)
	pl1467=(pl467*lf+pl167*lf+pl147*lf+pl146*lf)/(kd1+kd4+kd6+kd7)
	pl1567=(pl567*lf+pl167*lf+pl157*lf+pl156*lf)/(kd1+kd5+kd6+kd7)
	pl2345=(pl345*lf+pl245*lf+pl235*lf+pl234*lf)/(kd2+kd3+kd4+kd5)
	pl2346=(pl346*lf+pl246*lf+pl236*lf+pl234*lf)/(kd2+kd3+kd4+kd6)
	pl2347=(pl347*lf+pl247*lf+pl237*lf+pl234*lf)/(kd2+kd3+kd4+kd7)
	pl2356=(pl356*lf+pl256*lf+pl236*lf+pl235*lf)/(kd2+kd3+kd5+kd6)
	pl2357=(pl357*lf+pl257*lf+pl237*lf+pl235*lf)/(kd2+kd3+kd5+kd7)
	pl2367=(pl367*lf+pl267*lf+pl237*lf+pl236*lf)/(kd2+kd3+kd6+kd7)
	pl2456=(pl456*lf+pl256*lf+pl246*lf+pl245*lf)/(kd2+kd4+kd5+kd6)
	pl2457=(pl457*lf+pl257*lf+pl247*lf+pl245*lf)/(kd2+kd4+kd5+kd7)
	pl2467=(pl467*lf+pl267*lf+pl247*lf+pl246*lf)/(kd2+kd4+kd6+kd7)
	pl2567=(pl567*lf+pl267*lf+pl257*lf+pl256*lf)/(kd2+kd5+kd6+kd7)
	pl3456=(pl456*lf+pl356*lf+pl346*lf+pl345*lf)/(kd3+kd4+kd5+kd6)
	pl3457=(pl457*lf+pl357*lf+pl347*lf+pl345*lf)/(kd3+kd4+kd5+kd7)
	pl3467=(pl467*lf+pl367*lf+pl347*lf+pl346*lf)/(kd3+kd4+kd6+kd7)
	pl3567=(pl567*lf+pl367*lf+pl357*lf+pl356*lf)/(kd3+kd5+kd6+kd7)
	pl4567=(pl567*lf+pl467*lf+pl457*lf+pl456*lf)/(kd4+kd5+kd6+kd7)
	pl12345=(pl2345*lf+pl1345*lf+pl1245*lf+pl1235*lf+pl1234*lf)/(kd1+kd2+kd3+kd4+kd5)
	pl12346=(pl2346*lf+pl1346*lf+pl1246*lf+pl1236*lf+pl1234*lf)/(kd1+kd2+kd3+kd4+kd6)
	pl12347=(pl2347*lf+pl1347*lf+pl1247*lf+pl1237*lf+pl1234*lf)/(kd1+kd2+kd3+kd4+kd7)
	pl12356=(pl2356*lf+pl1356*lf+pl1256*lf+pl1236*lf+pl1235*lf)/(kd1+kd2+kd3+kd5+kd6)
	pl12357=(pl2357*lf+pl1357*lf+pl1257*lf+pl1237*lf+pl1235*lf)/(kd1+kd2+kd3+kd5+kd7)
	pl12367=(pl2367*lf+pl1367*lf+pl1267*lf+pl1237*lf+pl1236*lf)/(kd1+kd2+kd3+kd6+kd7)
	pl12456=(pl2456*lf+pl1456*lf+pl1256*lf+pl1246*lf+pl1245*lf)/(kd1+kd2+kd4+kd5+kd6)
	pl12457=(pl2457*lf+pl1457*lf+pl1257*lf+pl1247*lf+pl1245*lf)/(kd1+kd2+kd4+kd5+kd7)
	pl12467=(pl2467*lf+pl1467*lf+pl1267*lf+pl1247*lf+pl1246*lf)/(kd1+kd2+kd4+kd6+kd7)
	pl12567=(pl2567*lf+pl1567*lf+pl1267*lf+pl1257*lf+pl1256*lf)/(kd1+kd2+kd5+kd6+kd7)
	pl13456=(pl3456*lf+pl1456*lf+pl1356*lf+pl1346*lf+pl1345*lf)/(kd1+kd3+kd4+kd5+kd6)
	pl13457=(pl3457*lf+pl1457*lf+pl1357*lf+pl1347*lf+pl1345*lf)/(kd1+kd3+kd4+kd5+kd7)
	pl13467=(pl3467*lf+pl1467*lf+pl1367*lf+pl1347*lf+pl1346*lf)/(kd1+kd3+kd4+kd6+kd7)
	pl13567=(pl3567*lf+pl1567*lf+pl1367*lf+pl1357*lf+pl1356*lf)/(kd1+kd3+kd5+kd6+kd7)
	pl14567=(pl4567*lf+pl1567*lf+pl1467*lf+pl1457*lf+pl1456*lf)/(kd1+kd4+kd5+kd6+kd7)
	pl23456=(pl3456*lf+pl2456*lf+pl2356*lf+pl2346*lf+pl2345*lf)/(kd2+kd3+kd4+kd5+kd6)
	pl23457=(pl3457*lf+pl2457*lf+pl2357*lf+pl2347*lf+pl2345*lf)/(kd2+kd3+kd4+kd5+kd7)
	pl23467=(pl3467*lf+pl2467*lf+pl2367*lf+pl2347*lf+pl2346*lf)/(kd2+kd3+kd4+kd6+kd7)
	pl23567=(pl3567*lf+pl2567*lf+pl2367*lf+pl2357*lf+pl2356*lf)/(kd2+kd3+kd5+kd6+kd7)
	pl24567=(pl4567*lf+pl2567*lf+pl2467*lf+pl2457*lf+pl2456*lf)/(kd2+kd4+kd5+kd6+kd7)
	pl34567=(pl4567*lf+pl3567*lf+pl3467*lf+pl3457*lf+pl3456*lf)/(kd3+kd4+kd5+kd6+kd7)
	pl123456=(pl23456*lf+pl13456*lf+pl12456*lf+pl12356*lf+pl12346*lf+pl12345*lf)/(kd1+kd2+kd3+kd4+kd5+kd6)
	pl123457=(pl23457*lf+pl13457*lf+pl12457*lf+pl12357*lf+pl12347*lf+pl12345*lf)/(kd1+kd2+kd3+kd4+kd5+kd7)
	pl123467=(pl23467*lf+pl13467*lf+pl12467*lf+pl12367*lf+pl12347*lf+pl12346*lf)/(kd1+kd2+kd3+kd4+kd6+kd7)
	pl123567=(pl23567*lf+pl13567*lf+pl12567*lf+pl12367*lf+pl12357*lf+pl12356*lf)/(kd1+kd2+kd3+kd5+kd6+kd7)
	pl124567=(pl24567*lf+pl14567*lf+pl12567*lf+pl12467*lf+pl12457*lf+pl12456*lf)/(kd1+kd2+kd4+kd5+kd6+kd7)
	pl134567=(pl34567*lf+pl14567*lf+pl13567*lf+pl13467*lf+pl13457*lf+pl13456*lf)/(kd1+kd3+kd4+kd5+kd6+kd7)
	pl234567=(pl34567*lf+pl24567*lf+pl23567*lf+pl23467*lf+pl23457*lf+pl23456*lf)/(kd2+kd3+kd4+kd5+kd6+kd7)
	pl1234567=(pl234567*lf+pl134567*lf+pl124567*lf+pl123567*lf+pl123467*lf+pl123457*lf+pl123456*lf)/(kd1+kd2+kd3+kd4+kd5+kd6+kd7)
	return {'pf':pf,'lf':lf, 'pl1':pl1, 'pl2':pl2, 'pl3':pl3, 'pl4':pl4, 'pl5':pl5, 'pl6':pl6, 'pl7':pl7, 'pl12':pl12, 'pl13':pl13, 'pl14':pl14, 'pl15':pl15, 'pl16':pl16, 'pl17':pl17, 'pl23':pl23, 'pl24':pl24, 'pl25':pl25, 'pl26':pl26, 'pl27':pl27, 'pl34':pl34, 'pl35':pl35, 'pl36':pl36, 'pl37':pl37, 'pl45':pl45, 'pl46':pl46, 'pl47':pl47, 'pl56':pl56, 'pl57':pl57, 'pl67':pl67, 'pl123':pl123, 'pl124':pl124, 'pl125':pl125, 'pl126':pl126, 'pl127':pl127, 'pl134':pl134, 'pl135':pl135, 'pl136':pl136, 'pl137':pl137, 'pl145':pl145, 'pl146':pl146, 'pl147':pl147, 'pl156':pl156, 'pl157':pl157, 'pl167':pl167, 'pl234':pl234, 'pl235':pl235, 'pl236':pl236, 'pl237':pl237, 'pl245':pl245, 'pl246':pl246, 'pl247':pl247, 'pl256':pl256, 'pl257':pl257, 'pl267':pl267, 'pl345':pl345, 'pl346':pl346, 'pl347':pl347, 'pl356':pl356, 'pl357':pl357, 'pl367':pl367, 'pl456':pl456, 'pl457':pl457, 'pl467':pl467, 'pl567':pl567, 'pl1234':pl1234, 'pl1235':pl1235, 'pl1236':pl1236, 'pl1237':pl1237, 'pl1245':pl1245, 'pl1246':pl1246, 'pl1247':pl1247, 'pl1256':pl1256, 'pl1257':pl1257, 'pl1267':pl1267, 'pl1345':pl1345, 'pl1346':pl1346, 'pl1347':pl1347, 'pl1356':pl1356, 'pl1357':pl1357, 'pl1367':pl1367, 'pl1456':pl1456, 'pl1457':pl1457, 'pl1467':pl1467, 'pl1567':pl1567, 'pl2345':pl2345, 'pl2346':pl2346, 'pl2347':pl2347, 'pl2356':pl2356, 'pl2357':pl2357, 'pl2367':pl2367, 'pl2456':pl2456, 'pl2457':pl2457, 'pl2467':pl2467, 'pl2567':pl2567, 'pl3456':pl3456, 'pl3457':pl3457, 'pl3467':pl3467, 'pl3567':pl3567, 'pl4567':pl4567, 'pl12345':pl12345, 'pl12346':pl12346, 'pl12347':pl12347, 'pl12356':pl12356, 'pl12357':pl12357, 'pl12367':pl12367, 'pl12456':pl12456, 'pl12457':pl12457, 'pl12467':pl12467, 'pl12567':pl12567, 'pl13456':pl13456, 'pl13457':pl13457, 'pl13467':pl13467, 'pl13567':pl13567, 'pl14567':pl14567, 'pl23456':pl23456, 'pl23457':pl23457, 'pl23467':pl23467, 'pl23567':pl23567, 'pl24567':pl24567, 'pl34567':pl34567, 'pl123456':pl123456, 'pl123457':pl123457, 'pl123467':pl123467, 'pl123567':pl123567, 'pl124567':pl124567, 'pl134567':pl134567, 'pl234567':pl234567, 'pl1234567':pl1234567}