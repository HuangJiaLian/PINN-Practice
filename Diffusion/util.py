import numpy as np 
from numpy.linalg import inv

def get_num_solution(nx,nt,c):
    pi = np.pi
    deltax=1./(nx - 1)
    deltat=1./(nt - 1)
    r=deltat/deltax**2

    u = np.zeros([nt, nx])
    u[:,0]=1
    q = [0]*nx

    for i in range(nx):
        u[i,0] = 1
        q[i]=c*deltat*(np.sin(2.0*pi*i*deltax))
    p=1/2*np.diag(q)

    A = [0]*2
    B = [0]*2
    C = [0]*2
    D = [0]*2

    for i in range(2):
        if i == 0:
            A[i]=np.ones([1,nx])*(1+r)
            B[i]=np.ones([1,nx-1])*(-r/2)
            C[i]=np.ones([1,2])*(-r/2)    
        else:
            A[i]=np.ones([1,nx])*(1-r)
            B[i]=np.ones([1,nx-1])*(r/2)
            C[i]=np.ones([1,2])*(r/2) 
        A[i] = A[i].ravel()
        B[i] = B[i].ravel()
        C[i] = C[i].ravel()
        if i == 0:  
            D[i] = np.diag(A[i]) + np.diag(B[i],1) + np.diag(B[i],-1) + \
                np.diag(C[i],nx-2) + np.diag(C[i],-nx+2) + p 
        else:
            D[i] = np.diag(A[i]) + np.diag(B[i],1) + np.diag(B[i],-1) + \
                np.diag(C[i],nx-2) + np.diag(C[i],-nx+2) - p    
        D[i][1,nx-1]=0
        D[i][nx-2,0]=0   

    for i in range(nt-1):
        u[:,i+1] = np.dot(np.dot(inv(D[0]),D[1]),u[:,i])
    return u 