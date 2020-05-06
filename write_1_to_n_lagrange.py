"""
Write 1:n Lagrange ligand binding functions

When run, will generate lagrange-multiplier solved solutions to 1:n
protein-ligand binding.
"""

def tuple_to_balance(t):
    #pl123=(pl12*lf+pl13*lf+pl23*lf)/(kd1+kd2+kd3)
    get_tuple_without_value=lambda x,y : [z for z in x if z !=y]
    plwithoutone=combinations(t,len(t)-1)
    output=f"pl{''.join([str(x) for x in t])}=("
    for tv in t:
        output+="pl"
        a=get_tuple_without_value(t,tv)
        for v in a:
            output+=str(v)
        output+=f"*lf+"
    output=output[:-1]+")/("
    output+="+".join(["kd"+str(i) for i in t])+")\n"
    if len(t)==1:
        output=output.replace("pl", "pf")
    if output[0:2]=="pf":
        output[0:2]="pl"
    return output

from itertools import combinations
def write_1_to_n_lagrange(output_filename, n):
    out=open(output_filename,"w")
    out.write(r'"""'+"\n")
    out.write(f"1:{n} binding system solved using Lagrange multiplier approach\n")
    out.write("Modified Factory example utilising Lagrane multiplier to solve complex\n")
    out.write(f"concentration in a 1:{n} protein:ligand binding system\n")
    out.write(r'"""'+"\n")
    out.write("\nfrom timeit import default_timer as timer\nfrom scipy.optimize import fsolve\nimport autograd.numpy as np\nfrom autograd import grad, jacobian\n")
    out.write("def lagrange_1_to_"+str(n)+"(p0, l0,"+", ".join(['kd'+str(x) for x in range(1,n+1)])+"):\n")
    out.write("\tdef F(X): # Augmented Lagrange function\n")
    out.write("\t\tpf=X[0]\n")
    out.write("\t\tlf=X[1]\n")
    for i in range(1,n+1):
        out.write(f"\t\tpl{i}=pf*lf/kd{i}\n")
    bmat=list([list(combinations(range(1,n+1),x)) for x in range(1,n+1)])
    print(bmat)
    for i in range(1,len(bmat)):
        for j in range(len(bmat[i])):
            out.write("\t\t"+tuple_to_balance(bmat[i][j]))
    
    out.write(f"\t\tconstraint1=p0-(pf+")
    constraint=""
    for i in range(0,n):
        constraint+=f"{'+'.join(['pl'+''.join(str(b) for b in x) for x in bmat[i]])}+"
    constraint=constraint[:-1]+")\n"
    out.write(constraint)

    out.write(f"\t\tconstraint2=l0-(lf+")
    constraint=""
    for i in range(0,n):
        constraint+=f"{str(i+1)}*({'+'.join(['pl'+''.join(str(b) for b in x) for x in bmat[i]])})+"
    constraint=constraint[:-1]+")\n"
    out.write(constraint)
    out.write("\t\tnonzero_constraint=(constraint1-abs(constraint1))-(constraint2-abs(constraint2))\n")
    out.write(f"\t\treturn pl{''.join([str(x) for x in bmat[-1][0]])} - X[2]*constraint1 - X[3]*constraint2 - X[4]*nonzero_constraint\n")
    out.write("\tdfdL = grad(F, 0) # Gradients of the Lagrange function\n")
    out.write("\tpf, lf, lam1, lam2,lam3= fsolve(dfdL, [p0, l0]+[1.0]*3, fprime=jacobian(dfdL))\n")
    for i in range(1,n+1):
        out.write(f"\tpl{i}=pf*lf/kd{i}\n")
    for i in range(1,len(bmat)):
        for j in range(len(bmat[i])):
            out.write("\t"+tuple_to_balance(bmat[i][j]))
    out.write("\treturn {'pf':pf,'lf':lf")
    
    for i in bmat:
        for j in i:
            pl_line="pl"+"".join([str(s) for s in j])
            out.write(', \''+pl_line+'\':'+pl_line)
    out.write("}")

if __name__ == "__main__":
    concentrations=write_1_to_n_lagrange("1to10_lagrange.py", 5)
