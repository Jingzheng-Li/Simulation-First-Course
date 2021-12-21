
import taichi as ti
import numpy as np

ti.init(arch=ti.cpu, debug = False)

dim = 2
eps1, eps2 = 1e-6, 1e-6
mu1, mu2 = 0.5, 0.4
KMAX = 50
num_particles = 1
u0 = ti.Vector([1.0,1.0])

u = ti.Vector.field(dim,dtype=float,shape=num_particles)
M = ti.Matrix.field(dim,dim,dtype=float,shape=num_particles)
b = ti.Vector.field(dim,dtype=float,shape=num_particles)
dx = ti.Vector.field(dim,dtype=float,shape=num_particles)


@ti.func
def initial_stepsize(s1:float, s2:float):
    s = 0.0
    if s2 < s1: 
        s=s2
    else: 
        s=ti.min(1.0, 2.0*s2)
    return s


@ti.kernel
def damped_newton():
    for i in range(num_particles):  
        u[i] = u0
        s1, s2 = 1.0, 1.0
        norm_du2 = 1.0
        for k in range(KMAX):
            b[i] = TestFunction(u[i].x, u[i].y)
            norm_b = b[i].norm()
            if norm_b < eps1:
                print(k, ": norm_Fu=", norm_b," norm_du=",norm_du2," alpha=", s2, " u[i]=",u[i])
                break 
            M[i] = Jacobian(u[i].x, u[i].y)
            dx[i] = -M[i].inverse() @ b[i]
            s = initial_stepsize(s1,s2);
                      
            #armijo line search steps
            Fx = TestFunction((u[i]+s*dx[i]).x,(u[i]+s*dx[i]).y)
            norm_Fx = Fx.norm()
            while norm_Fx>(1-mu1*s)*norm_b:
                s = 0.5*s
                Fx = TestFunction((u[i]+s*dx[i]).x,(u[i]+s*dx[i]).y)
                norm_Fx = Fx.norm()
            s1 = s2
            s2 = s

            u[i] = u[i] + s2 * dx[i]
            norm_du = dx[i].norm()
            if norm_du < eps2:
                print(k, ": norm_Fu=", norm_b," norm_du=",norm_du2," alpha=", s2, " u[i]=",u[i])
                break
            print(k, ": norm_Fu=", norm_b," norm_du=",norm_du2," alpha=", s2, " u[i]=",u[i])


#==================================================================
#only need to change TestFunction and Jacobian
@ti.func
def TestFunction(ux:float,uy:float):
    b = ti.Vector([0.0,0.0])
    b.x = ux + 3*ti.log(ti.abs(ux)) - uy**2
    b.y = 2*ux**2 - ux*uy - 5*ux + 1
    return b

@ti.func
def Jacobian(ux:float,uy:float):
    Jac = ti.Matrix([[1+3/ux, -2*uy],[4*ux-uy-5,-ux]])
    return Jac

damped_newton()
print(u[0])



