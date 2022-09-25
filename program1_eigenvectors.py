import numpy as np
import matplotlib.pyplot as plt

#!This works only for 2D matrices.!
#change m here

m = [[1,4],[2,3]]

def second_dim_eigen(m):
    a = np.linspace(0,1,100)
    eig_val_cov,eig_vec_cov = np.linalg.eig(m)

    #eigenvectors
    E1 = [eig_vec_cov[0][0]*x/eig_vec_cov[1][0] for x in a]
    E2 = [eig_vec_cov[0][1]*x/eig_vec_cov[1][1] for x in a]
    
    eigen=[]
    for i in range(len(eig_vec_cov)):
        eigen.append(eig_vec_cov[:,i])
    #eigenvectors multiplied with the matrix m
    ae1 = np.matmul(m,eigen[0])
    ae2 = np.matmul(m,eigen[1])
    set1 = [ae1[0]*x/ae1[1] for x in a]
    set2 = [ae2[0]*x/ae2[1] for x in a]

    #plot m1 and m2
    vec1 = [m[0][0]*x/m[0][1] for x in a]
    vec2 = [m[1][0]*x/m[1][1] for x in a]



    plt.plot(a,E1,'y',label='E-vector 1') 
    plt.plot(a,E2,'c',label="E-vector 2")
    plt.plot(a,vec1,lw=1.,ls='--',c='k',label="A1")
    plt.plot(a,vec2,ls='--',c='b',label="A2")
    plt.legend()
    plt.xlabel("A1,A2 are given vectors; E vectors are eigen vectors")
    plt.show()

    plt.plot(a,E1,'y',label='E-vector 1') 
    plt.plot(a,E2,'c',label="E-vector 2")
    plt.plot(a,set1,ls='--',c='k',label="A*v1")
    plt.plot(a,set2,ls='--',c='r',label = "A*v2")
    plt.legend()
    plt.xlabel("v1 = E-vector1, v2 = E-vector2")
    plt.show()
second_dim_eigen(m)
