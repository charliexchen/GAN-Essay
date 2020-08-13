import numpy as np
import math
class Q:
    #define rational number
    def __init__(self, num, denum):
        self.num=int(num)
        self.denum=int(denum)
        self.cancel()
        #initiate values
    def va(self):
        return float(self.num)/self.denum
        #evaluate values
    def __repr__(self):
        #return str(self.va())
        return str([self.num, self.denum])
    def __str__(self):
        return str([self.num, self.denum])
        #returns the numerator and denominator when printed
    def __add__(self, other):
        return Q(self.num*other.denum+other.num*self.denum, self.denum*other.denum)
    def __sub__(self, other):
        return Q(self.num*other.denum-other.num*self.denum, self.denum*other.denum)
    def __mul__(self, other):
        return Q(self.num*other.num, self.denum*other.denum)
    def __div__(self, other):
        return Q(self.num*other.denum, self.denum*other.num)
    def __neg__(self):
        return Q(-self.num, self.denum)
    def __eq__(self,other):
        if self.num*other.denum-other.num*self.denum==0:
            return True
        else:
            return False
    #overloads all field operations
    def cancel(self):
        a=self.num
        b=self.denum
        while b!=0:
            remainder = a%b
            a=b
            b=remainder
        #runs Euclid's algorithm
        #a is now the gcd of num and denu,
        self.num=self.num/a
        self.denum=self.denum/a
    #makes sure that fraction is in simplest form


def subt (eq,i,j,l):
    #subtract l copies of the ith row from the jth row
    A=eq[0]
    b=eq[1]
    b[j]=b[j]-l*b[i]
    for k in range (0,len(A[i])):
        A[j][k]=A[j][k]-l*A[i][k]
    return (A,b)

def triQ(eq):
    pass
def GaussQ(eq):
    Ab=eq
    for x in range (0,len(eq[0][0])):
        for y in range (1+x,len(eq[0])):
            if (Ab[0][x][x]==Q(0,1)):
                print 'error!'
            Ab=subt(Ab,x,y,Ab[0][y][x]/Ab[0][x][x])

            #gaussian elim to obtain triangluar matrix
    for x in range (0,len(eq[0][0])):
        for y in range (1+x,len(eq[0])):
            Ab=subt(Ab,len(eq[0][0])-1-x,len(eq[0][0])-1-y,
            Ab[0][len(eq[0][0])-1-y][len(eq[0][0])-1-x]/Ab[0][len(eq[0][0])-1-x][len(eq[0][0])-1-x])
            #further elim to obtain diagonal matrix

    x=[]
    for a in range (0,len(eq[1])):
        x+=[Ab[1][a]/Ab[0][a][a]]
        #obtain x from matrix
    return x
    #return x of Ax=b

def p(b,n):
    #calculates power series of (x+1)^b at x=0
    c=Q(1,1)
    power=Q(b,1)
    for i in range (1, n+1):
        c=(c*power)
        if i>0:
            c=c/Q(i,1)
        power=power-Q(1,1)
    return c
def f1(n):
    return p(Q(1,2), n)

def f4(n):
    return Q(1,float(math.factorial(n)))
def f5(n):
    c=Q(0,1)
    for i in range (0,n+1):
        c+=f4(i)*p(-1,n-i)
    return c
def padeQ(c, L,eva=False):
    #returns the [L,M] Pade approximant coeffecients
    #c is an array of of power series terms for some function.
    #it and tol are the tolerances of the iterative improvement solver, as defined in iterImprovSolver
    #eval setting eval as a float evaluates pade approximant at that position
    M1=[]
    b1=[]
    for r in range(L+1,2*L+1):
        b1=b1+[-c[r]]
        row=[]
        for k in range(1, L+1):
            if k<=min(r,L):
                row+=[c[r-k]]
            else:
                row+=[0]
        M1=M1+[row]
    #define a matrix M1, vector b1 for our first set of simultanious equations
    q=GaussQ([M1,b1])
    q=[Q(1,1)]+q
    #set q[0]=1 to correct the indices
    p=[]
    for k in range (0, L+1):
        co=c[k]
        for s in range(1, min(k,L)+1):
            co+=c[k-s]*q[s]
        p=p+[co]
    
    #if eva has a value, evaluate the approximant at that position, otherwise just return the coefficients
    if eva:
        num=0
        den=0
        for k in range (0,max(L,L)+1):
            if k<=L:
                num+=p[k]*(eva**k)
            if k<=M:
                den+=q[k]*(eva**k)
        return num/den
    else:
        return [p,q]
def pade_funcQ(func, L):
    #use this when we have a function output func for the power series
    #all other parameters same as pade()
    c=[]
    for a in range (0,2*L+1):
        c=c+[func(a)]
    return padeQ (c,L)


def rootfinder(p):
    #returns roots in array
    #p[i] is the ith power element of the array
    poly=[]
    for a in range(0, len(p)):
        poly=[p[a].va()]+poly
    #np.roots takes inputs with the largest power first, so we need to reverse the list p
    return np.roots(poly)

print rootfinder(pade_funcQ(f5,12)[1])
