import pandas as pd
import matplotlib.pyplot as pp
from numpy import *

def cf(p,y,m):
	cost=sum(square(p-y))/(2*m)	
	return cost

def grad(p,y,m,n,x,alp,th):
	l=[]
	for i in range(n):
                z=x[:,i].reshape(m,1)
                v=sum((p-y)*z)
                l.append(th[i]-(alp*v/m))	
	
	for i in range(n):
		th[i]=l[i]
	
	return th
	

data=pd.read_csv("ex1data1.csv")
m=len(data)
n=2
d=array(data).reshape(m,2)
x1=d[:,0].reshape(m,1)
y=d[:,1].reshape(m,1)

o=ones((m,1),int)
x=hstack((o,x1));

th=array([[0.0],[0.0]]).reshape(n,1)

p=dot(x,th)

cost=cf(p,y,m)
print("Initial Cost: {}".format(cost))

#pp.plot(x,p)
#pp.show()

it=10000
itr=range(it)
alp=0.01
j=[]

for i in range(it):
	th=grad(p,y,m,n,x,alp,th)
	p=dot(x,th)
	j.append(cf(p,y,m))
	
	
print("Final Cost: {}".format(j[it-1]))
print("Parameter Vector: {}".format(th))
print("Graph Show....")

pp.plot(itr[0:it],j[0:it])
pp.title("Cost vs Itration Plot")
pp.xlabel("Iteration")
pp.ylabel("Loss")
pp.show()



	

