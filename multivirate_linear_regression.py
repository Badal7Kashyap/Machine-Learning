import pandas as pd 
import matplotlib.pyplot as pp
from numpy import *

def cf(p,y,m):
	return sum(square(p-y))/(2*m)

def grad(p,y,m,n,x,alp):
	l=[]
	for i in range(n):
		z=x[:,i].reshape(m,1)
		v=sum((p-y)*z)
		l.append(th[i]-(alp*v/m))

	for i in range(n):
		th[i]=l[i]

	return th


#read dataset
data=pd.read_csv("ex1data2.csv")
sz=len(data)
n=3
dt=array(data).reshape(sz,n)

#training data
m=30
d=dt[0:m,:].reshape(m,n)
x1=d[:,0].reshape(m,1)
x2=d[:,1].reshape(m,1)
y=d[:,2].reshape(m,1)
#print(d)

#testing data
m1=sz-m
td=dt[m:,:].reshape(m1,n)
tx1=td[:,0].reshape(m1,1)
tx2=td[:,1].reshape(m1,1)
ty=td[:,2].reshape(m1,1)
#print(td)



#feature scalling
mx1=mean(x1)
mx2=mean(x2)
sd1=std(x1)
sd2=std(x2)

x1=(x1-mx1)/sd1
x2=(x2-mx2)/sd2
x0=ones((m,1),int)

x=hstack((x0,x1,x2))

#prediction
th=array([[0.0],[0.0],[0.0]]).reshape(n,1)
p=dot(x,th)

#loss
cost=cf(p,y,m)
print("Initial Cost: {}".format(cost))

#gradient descent
it=500
itr=range(it)
j=[]
alp=0.1

for i in range(it):
	th=grad(p,y,m,n,x,alp)
	p=dot(x,th)
	j.append(cf(p,y,m))

print("Final Cost: {}".format(j[it-1]))
print("Parameters : {}".format(th))

#prediction
print("|||| PREDICTION TESTING DATASET ||||")
print("")
for i in range(m1):
	v1=(tx1[i]-mx1)/sd1
	v2=(tx2[i]-mx2)/sd2
	result=th[0]+(th[1]*v1)+(th[2]*v2)
	print("Predicted Price: ",int(result),end=" || ")
	print("Actual Price: ",int(ty[i]))

	
#custom input
print("")
print("Custom Input")
a=input("y/n: ")
if(a=="y"):
	f1=int(input("Area(sqr^mtrs) : "))
	f2=int(input("No Of Bedrooms : "))

	v1=(f1-mx1)/sd1
	v2=(f2-mx2)/sd2

	result=th[0]+(th[1]*v1)+(th[2]*v2)		
	print("Price: {}".format(int(result)))
	print("")
#plotting
print("Graph....")
print("| J(theta) vs Iteration |")
pp.plot(itr[0:it],j[0:it])
pp.title("Cost vs Iteration")
pp.xlabel("Iteration")
pp.ylabel("Loss")
pp.show()



	



