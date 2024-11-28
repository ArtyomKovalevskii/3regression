import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.integrate as integrate
from scipy.integrate import quad, dblquad
from numpy import sqrt, sin, cos, pi, exp, log
from scipy.special import factorial, binom, gamma, loggamma, zeta, gammainc, legendre, eval_legendre
import scipy.special as special
from numpy import linalg as LA


n=200
N=20000

def regressor_1(n):
    X=np.random.standard_normal((n,3))
    X[:,0]=1.
    return X

def regressor_2(n):
    X=np.random.standard_normal((n,3))
    X[:,0]=1.
    X[:,2]+=3*X[:,2]
    return X

def data_true(regressor,n):
    points=np.zeros((n,4))
    points[:,1:4]=regressor
    points[:,0]=np.random.standard_normal(n)
    points[:,0]+=points[:,1]+points[:,2]+points[:,3]
    Y=points[:,0]
    return points, regressor, Y

def data_false_3(regressor,n):
    points=np.zeros((n,4))
    points[:,1:4]=regressor
    points[:,0]=np.random.standard_normal(n)
    points[:,0]+=points[:,1]+points[:,2]+points[:,3]
    for i in range(n):
        if points[i,2]>=0:
            points[i,0]+=1
    Y=points[:,0]
    return points, regressor, Y

def data_false_4(regressor,n):
    points=np.zeros((n,4))
    points[:,1:4]=regressor
    points[:,0]=np.random.standard_normal(n)
    points[:,0]+=points[:,1]+points[:,2]+points[:,3]
    for i in range(n):
        if points[i,2]+points[i,3]>=0:
            points[i,0]+=1
    Y=points[:,0]
    return points, regressor, Y




def theta_hat(X,Y):
    t=np.matmul(np.linalg.inv(np.matmul(X.T,X)),
                                 np.matmul(X.T,Y))
    return t

def epsilon_hat(X,Y,theta,n):
    return Y-np.matmul(X,theta)

def sigma_tilde(epsilon):
    sigma_tilde=np.std(epsilon)
    return sigma_tilde

def epsilon_hat_j(X,Y,j,theta,n):
    points_sorted=points[np.argsort(points[:,j])]
    Y_sorted=points_sorted[:,0]
    X_sorted=points_sorted[:,1:4]
    epsilon_hat_j=epsilon_hat(X_sorted,Y_sorted,theta,n)
    return epsilon_hat_j

def Delta_j(epsilon_j):
    Delta_j=np.cumsum(epsilon_j,axis=0)
    return Delta_j

def omega_2_n(Delta_2,Delta_3,sigma,n):
    omega_2_n=0
    for j in range(n-1):
        omega_2_n+=Delta_2[j]*(2*Delta_2[j]+Delta_2[j+1])
        omega_2_n+=Delta_3[j]*(2*Delta_3[j]+Delta_3[j+1])
    omega_2_n=omega_2_n/3/n**2/sigma**2
    return omega_2_n


def L_2(t,X,n):
    L_2=np.zeros(3)
    X_sorted=X[np.argsort(X[:,1])]
    L_2[0]=t
    L_2[1]=1/n*np.sum(X_sorted[:int(np.floor(n*t)),1])
    L_2[2]=1/n*np.sum(X_sorted[:int(np.floor(n*t)),2])
    return L_2

def L_3(t,X,n):
    L_3=np.zeros(3)
    X_sorted=X[np.argsort(X[:,2])]
    L_3[0]=t
    L_3[1]=1/n*np.sum(X_sorted[:int(np.floor(n*t)),1])
    L_3[2]=1/n*np.sum(X_sorted[:int(np.floor(n*t)),2])
    return L_3

def G_n(X):
    G_n=np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            G_n[i,j]=np.average(X[:,i]*X[:,j])
    return G_n

def K_22(s,t,X,G,n):
    K_22=min(s,t)
    K_22-=np.matmul(np.matmul(L_2(s,X,n),np.linalg.inv(G)), L_2(t,X,n).T)
    return K_22

def K_33(s,t,X,G,n):
    K_33=min(s,t)
    K_33-=np.matmul(np.matmul(L_3(s,X,n),np.linalg.inv(G)), L_3(t,X,n).T)
    return K_33

def K_23(s,t,X,G,n):
    X_sorted_2=X[np.argsort(X[:,1])]
    if s==1:
        X_s=X_sorted_2[-1,1]
    if s<1:
        X_s=X_sorted_2[int(np.floor(n*s)),1]
    X_sorted_3=X[np.argsort(X[:,2])]
    if t==1:
        X_t=X_sorted_3[-1,2]
    if t<1:
        X_t=X_sorted_3[int(np.floor(n*t)),2]
    K_23=0
    for i in range(n):
        if X[i,1]<=X_s:
            if X[i,2]<=X_t:
                K_23+=1
    K_23=K_23/n-np.matmul(np.matmul(L_2(s,X,n),np.linalg.inv(G)), L_3(t,X,n).T)
    return K_23

def K_tilde(s,t,X,G,n):
    K_tilde=0
    if s<=1:
        if t<=1:
            K_tilde=K_22(s,t,X,G,n)
        if t>1:
            K_tilde=K_23(s,t-1,X,G,n)
    if s>1:
        if t<=1:
            K_tilde=K_23(t,s-1,X,G,n)
        if t>1:
            K_tilde=K_33(s-1,t-1,X,G,n)
    return K_tilde


def R_matrix(X,G,n,M):
    R_matrix=np.zeros((M,M))
    for i in range(M):
        for j in range(M):
            R_matrix[i,j]=dblquad(lambda s,t: K_tilde(s,t,X,G,n)*np.sin(np.pi*(i+1)*s/2)*np.sin(np.pi*(j+1)*t/2), 0, 2, lambda s: 0, lambda s: 2)
            #print('yes')
    return R_matrix
            
n_points=50
step=2/n_points

def integrand(s,t,X,G,n,i,j):
    return K_tilde(s,t,X,G,n)*np.sin(np.pi*(i+1)*s/2)*np.sin(np.pi*(j+1)*t/2)

def R_m(X,G,n,M):
    R_m=np.zeros((M,M))
    for i in range(M):
        for j in range(M):
            for i1 in range(n_points):
                for j1 in range(n_points):
                    R_m[i,j]+=integrand(i1*step,j1*step,X,G,n,i,j)*step**2
            print('yes')
    return R_m
            
def Determ(lam,lambda_list):
        kk=math.floor(len(lambda_list)/2)
        D=1
        for k in range(1,2*kk+1):
                D=D*(1-lam/lambda_list[k-1])
        return D
        
def Smirnov(x,lambda_list):
        F=1
        kk=math.floor(len(lambda_list)/2)
        for k in range(1,kk+1):
                F=F+(-1)**k/pi*integrate.quad(lambda la: exp(-la*x/2)*(-Determ(la,lambda_list))**(-0.5)/la, lambda_list[2*k-2], lambda_list[2*k-1])[0]
        return F
        
            
################  Experiment 1


regressor=regressor_1(n)
omega_2_vec=np.zeros(N)

for ii in range(N):

    points,X,Y=data_true(regressor,n)
    #print('done')
    theta=theta_hat(X,Y)
    epsilon=epsilon_hat(X,Y,theta,n)
    sigma=sigma_tilde(epsilon)            
    epsilon_2=epsilon_hat_j(X,Y,2,theta,n)
    epsilon_3=epsilon_hat_j(X,Y,3,theta,n)
    Delta_2=Delta_j(epsilon_2)
    Delta_3=Delta_j(epsilon_3)
    omega_2=omega_2_n(Delta_2,Delta_3,sigma,n)
    omega_2_vec[ii]=omega_2

M=42

G=G_n(X)
R=R_m(X,G,n,M)

print(R)

eiv, v = LA.eigh(R)


print('eigenvalues')
print(eiv)


lambda_list=[]
for i in range(M-2):
        ll=1/eiv[M-i-1]
        lambda_list.append(ll)

print('lambda_list')

print(lambda_list)

t=[]
s=[]
for i in range(40):
    t.append(0.05+0.01*i)
    s.append(Smirnov(0.05+0.01*i,lambda_list))
    

#plt.plot(t,s)
#sns.ecdfplot(omega_2_vec)
#plt.grid()
#plt.show()

xx1=X
t1=t
s1=s
o1=omega_2_vec

################  Experiment 2


regressor=regressor_2(n)
omega_2_vec=np.zeros(N)

for ii in range(N):

    points,X,Y=data_true(regressor,n)
    #print('done')
    theta=theta_hat(X,Y)
    epsilon=epsilon_hat(X,Y,theta,n)
    sigma=sigma_tilde(epsilon)            
    epsilon_2=epsilon_hat_j(X,Y,2,theta,n)
    epsilon_3=epsilon_hat_j(X,Y,3,theta,n)
    Delta_2=Delta_j(epsilon_2)
    Delta_3=Delta_j(epsilon_3)
    omega_2=omega_2_n(Delta_2,Delta_3,sigma,n)
    omega_2_vec[ii]=omega_2

M=42

G=G_n(X)
R=R_m(X,G,n,M)

print(R)

eiv, v = LA.eigh(R)


print('eigenvalues')
print(eiv)


lambda_list=[]
for i in range(M-2):
        ll=1/eiv[M-i-1]
        lambda_list.append(ll)

print('lambda_list')

print(lambda_list)

t=[]
s=[]
for i in range(40):
    t.append(0.05+0.01*i)
    s.append(Smirnov(0.05+0.01*i,lambda_list))
    


xx2=X
t2=t
s2=s
o2=omega_2_vec



################  Experiment 3


regressor=regressor_1(n)
omega_2_vec=np.zeros(N)

for ii in range(N):

    points,X,Y=data_false_3(regressor,n)
    #print('done')
    theta=theta_hat(X,Y)
    epsilon=epsilon_hat(X,Y,theta,n)
    sigma=sigma_tilde(epsilon)            
    epsilon_2=epsilon_hat_j(X,Y,2,theta,n)
    epsilon_3=epsilon_hat_j(X,Y,3,theta,n)
    Delta_2=Delta_j(epsilon_2)
    Delta_3=Delta_j(epsilon_3)
    omega_2=omega_2_n(Delta_2,Delta_3,sigma,n)
    omega_2_vec[ii]=omega_2

M=42

G=G_n(X)
R=R_m(X,G,n,M)

print(R)

eiv, v = LA.eigh(R)


print('eigenvalues')
print(eiv)


lambda_list=[]
for i in range(M-2):
        ll=1/eiv[M-i-1]
        lambda_list.append(ll)

print('lambda_list')

print(lambda_list)

t=[]
s=[]
for i in range(40):
    t.append(0.05+0.01*i)
    s.append(Smirnov(0.05+0.01*i,lambda_list))
    


xx3=X
t3=t
s3=s
o3=omega_2_vec

################  Experiment 4


regressor=regressor_2(n)
omega_2_vec=np.zeros(N)

for ii in range(N):

    points,X,Y=data_false_4(regressor,n)
    #print('done')
    theta=theta_hat(X,Y)
    epsilon=epsilon_hat(X,Y,theta,n)
    sigma=sigma_tilde(epsilon)            
    epsilon_2=epsilon_hat_j(X,Y,2,theta,n)
    epsilon_3=epsilon_hat_j(X,Y,3,theta,n)
    Delta_2=Delta_j(epsilon_2)
    Delta_3=Delta_j(epsilon_3)
    omega_2=omega_2_n(Delta_2,Delta_3,sigma,n)
    omega_2_vec[ii]=omega_2

M=42

G=G_n(X)
R=R_m(X,G,n,M)

print(R)

eiv, v = LA.eigh(R)


print('eigenvalues')
print(eiv)


lambda_list=[]
for i in range(M-2):
        ll=1/eiv[M-i-1]
        lambda_list.append(ll)

print('lambda_list')

print(lambda_list)

t=[]
s=[]
for i in range(40):
    t.append(0.05+0.01*i)
    s.append(Smirnov(0.05+0.01*i,lambda_list))
    


xx4=X
t4=t
s4=s
o4=omega_2_vec

################## pictures

set1=[]
for i in range(len(xx1[:,0])):
    if xx1[i,1]>=0:
        set1.append([xx1[i,1],xx1[i,2],1])
    if xx1[i,1]<0:
        set1.append([xx1[i,1],xx1[i,2],2])
        
set2=[]
for i in range(len(xx2[:,0])):
    if xx2[i,1]+xx2[i,2]>=0:
        set2.append([xx2[i,1],xx2[i,2],1])
    if xx2[i,1]+xx2[i,2]<0:
        set2.append([xx2[i,1],xx2[i,2],2])

df1 = pd.DataFrame(set1, columns=['x1', 'x2', 'class'])
df2 = pd.DataFrame(set2, columns=['x1', 'x2', 'class'])


# Create a figure with 1 row and 2 columns
fig, axes = plt.subplots(1, 2, figsize=(18, 5))

# Plot on the first subplot
sns.scatterplot(data=df1, x="x1", y="x2", style="class", ax=axes[0])
axes[0].set_title('Experiments 1 and 3')

# Plot on the second subplot
sns.scatterplot(data=df2, x="x1", y="x2", style="class", ax=axes[1])
axes[1].set_title('Experiments 2 and 4')

# Adjust the spacing between plots
plt.tight_layout()

# Display the plot
plt.show()



fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(t1, s1)
sns.ecdfplot(o1, ax=axs[0,0])
axs[0, 0].grid()
axs[0, 0].set_title('Experiment 1')

axs[0, 1].plot(t2, s2)
sns.ecdfplot(o2, ax=axs[0,1])
axs[0, 1].grid()
axs[0, 1].set_title('Experiment 2')

axs[1, 0].plot(t3, s3)
sns.ecdfplot(o3, ax=axs[1,0])
axs[1, 0].grid()
axs[1, 0].set_title('Experiment 3')

axs[1, 1].plot(t4, s4)
sns.ecdfplot(o4, ax=axs[1,1])
axs[1, 1].grid()
axs[1, 1].set_title('Experiment 4')

fig.tight_layout()
plt.show()


