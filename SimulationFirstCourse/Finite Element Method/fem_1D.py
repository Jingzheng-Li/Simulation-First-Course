import matplotlib.pyplot as plt
import numpy as np
import taichi as ti

ti.init(arch=ti.gpu, debug=True)

#test the finite element method for 1D elliptic equaiton
#-u_xx + 0.25*pi^2*u = 0.5*pi^2*sin(0.5*pi*x) 0<x<1
#u(0) = 0, u'(1) = 0

c0 = 0.25 * np.pi**2;
alpha = 0.0;
beta = 0.0;
n = 10;
h = 1/n;
num_particles = 1

#这个等一下看看能不能放在taichi scope里面写
x = ti.Vector.field(n+1,dtype=float,shape=num_particles)
for i,j in ti.static(ti.ndrange(num_particles,n+1)):
    x[i][j] = h*j


#n+1*n+1 sparse matrix
K = ti.field(dtype=float)
ti.root.pointer(ti.ij,(n+1,n+1)).place(K)
M = ti.field(dtype=float)
ti.root.pointer(ti.ij,(n+1,n+1)).place(M)
A = ti.field(dtype=float)
ti.root.pointer(ti.ij,(n+1,n+1)).place(A)
B = ti.field(dtype=float)
ti.root.pointer(ti.i,n+1).place(B)
A_without_boundary = ti.field(dtype=float)
ti.root.pointer(ti.ij,(n,n)).place(A_without_boundary)
B_without_boundary = ti.field(dtype=float)
ti.root.pointer(ti.i,n).place(B_without_boundary)
init_x = B_without_boundary



def gradient_desc(A,b,init_x,iters=10):
    dim = init_x.shape[0]
    x = init_x
    result = np.zeros((iters+1,dim))
    result[0,:] = x.reshape(-1)
    for i in range(iters):
        ax = A @ x
        grad = ax - b
        flat_x = x.reshape(-1)
        lr = np.dot(flat_x,flat_x)/np.dot(ax.reshape(-1),flat_x)
        x -= grad * lr
        result[i+1,:] = x.reshape(-1)
    return result


@ti.kernel
def assemble_fem_p1():

    #assemble stiffness matrix
    for i in range(n):
        #this can put outside if mesh is even and stable
        Ke = 1/h * ti.Matrix([[1,-1],[-1,1]])
        for j,k in ti.static(ti.ndrange(2,2)):
            K[i+j,i+k] = K[i+j,i+k] + Ke[j,k]

    #assemble mass matrix
    for i in range(n):
        #this can put outside if mesh is even and stable
        Me = c0*h*ti.Matrix([[1/3,1/6],[1/6,1/3]])
        for j,k in ti.static(ti.ndrange(2,2)):
            M[i+j,i+k] = M[i+j,i+k] + Me[j,k]

    #assemble F matrix
    for i,j in ti.static(ti.ndrange(n,num_particles)):
        Be = h*ti.Vector([TestFunction(x[j][i])*(1/2),
                          TestFunction(x[j][i+1])*(1/2)])
        for k in ti.static(range(2)):
            B[i+k] = B[i+k] + Be[k]

    print("finished assemble step")


@ti.kernel
def fem_p1():

    #这一步倒是不难写 但是想写快感觉很麻烦
    for i,j in ti.static(ti.ndrange(n+1,n+1)):
        A[i,j] = K[i,j] + M[i,j]
        print(A[i,j])

    # handle boundary conditions
    #先这么写 等下看看如果先不删掉第一个 然后后面取下第一个会有多大影响
    #for i,j in ti.static(ti.ndrange(n,n)):
    #    A_without_boundary[i,j] = A[i+1,j+1]

    #for i,j in ti.static(ti.ndrange(n,n)):
    #    B_without_boundary[i] = B[i+1]

    #uh = gradient_desc(A_without_boundary,
    #                   B_without_boundary,
    #                   init_x)





@ti.func
def TestFunction(x:float):
    y = 0.5*np.pi**2*ti.sin(0.5*np.pi*x)
    return y

@ti.func
def ExactSolution(x:float):
    y = ti.sin(0.5*np.pi*x)
    return y


assemble_fem_p1()
fem_p1()




