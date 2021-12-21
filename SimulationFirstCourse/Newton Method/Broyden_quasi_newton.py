
import taichi as ti
import numpy as np

ti.init(arch=ti.cpu, debug = False)

dim = 2
eps1, eps2 = 1e-6, 1e-6
KMAX = 50
num_particles = 1
B0 = ti.Matrix([[1.0,0.0],[0.0,1.0]])
u0 = ti.Vector([1.0,1.0])

u = ti.Vector.field(dim,dtype=float,shape=num_particles)
B = ti.Matrix.field(dim,dim,dtype=float,shape=num_particles)
Fu1 = ti.Vector.field(dim,dtype=float,shape=num_particles)
Fu2 = ti.Vector.field(dim,dtype=float,shape=num_particles)
du = ti.Vector.field(dim,dtype=float,shape=num_particles)



@ti.kernel
def Broyden_quasi_newton():

    for i in range(num_particles):
        norm_du1 = 1.0
        Fu1[i] = TestFunction(u0.x, u0.y)
        B[i] = B0
        u[i] = u0
        for k in range(KMAX):
            du[i] = -B[i] @ Fu1[i]
            u[i] = u[i] + du[i]
            Fu2[i] = TestFunction(u[i].x,u[i].y)
            norm_Fu2 = Fu2[i].norm()
            norm_du2 = du[i].norm()
            ratio = norm_du2/norm_du1
            if norm_Fu2<eps1 or norm_du2<eps2:
                print(k, ": norm_Fu=", norm_Fu2," norm_du=",norm_du2," ratio=", ratio, " u[i]=",u[i])
                break   
            
            #broyden inverse update
            s = du[i]
            y = Fu2[i] - Fu1[i]
            numerator_tmp = (s-B[i]@y)@(s.transpose()@B[i])
            denominator_tmp = s.transpose()@B[i]@y #constant float
            B[i] = B[i] + numerator_tmp/denominator_tmp.x
            print(k, ": norm_Fu=", norm_Fu2," norm_du=",norm_du2," ratio=", ratio, " u[i]=",u[i])
            Fu1[i] = Fu2[i]
            norm_Fu1 = norm_Fu2
            norm_du1 = norm_du2


@ti.func
def TestFunction(ux:float,uy:float):
    b = ti.Vector([0.0,0.0])
    b.x = ux + 3*ti.log(ti.abs(ux)) - uy**2
    b.y = 2*ux**2 - ux*uy - 5*ux + 1
    return b

Broyden_quasi_newton()
print(u[0])



