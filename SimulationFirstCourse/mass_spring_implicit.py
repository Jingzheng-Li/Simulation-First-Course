

import taichi as ti
ti.init(arch=ti.gpu)


#最大粒子数目
max_num_particles = 1024
dt = 1e-3
substeps = 10
particle_mass = 1.0
attract_force = 100
gravity = ti.Vector([0,-9.8])



#force
spring_Y = ti.field(dtype=ti.f32,shape=())
drag_damping = ti.field(dtype=ti.f32,shape=())
dashpot_damping = ti.field(dtype=ti.f32,shape=())


#粒子的位置和速度场 xy两个分量
num_particles = ti.field(dtype=ti.i32,shape=())
x = ti.Vector.field(2,dtype=ti.f32,shape=max_num_particles)
v = ti.Vector.field(2,dtype=ti.f32,shape=max_num_particles)
f = ti.Vector.field(2,dtype=ti.f32,shape=max_num_particles)
fixed = ti.field(dtype=ti.i32,shape=max_num_particles)


#rest_length = 0:two particles are not connected
#rest_length != 0:two particles are connected with spring rest_length[i,j]
rest_length = ti.field(dtype=ti.f32, shape=(max_num_particles,max_num_particles))


# mass matrix diag(m1,m2,...,mn)
M = ti.Matrix.field(2,2,ti.f32,shape=(max_num_particles,max_num_particles))
# Jacobian matrix of partial_f/partial_x
J = ti.Matrix.field(2,2,ti.f32,shape= (max_num_particles,max_num_particles))
# A = [M-beta*dt*dt*J]
A = ti.Matrix.field(2,2,ti.f32,shape=(max_num_particles,max_num_particles))
#force vector
F = ti.Vector.field(2,ti.f32,shape=max_num_particles)
#force vector
b = ti.Vector.field(2,ti.f32,shape=max_num_particles)
#iteration temp variables
new_v = ti.Vector.field(2,ti.f32,shape=max_num_particles)

@ti.kernel
def update_mass_matrix():
    m = ti.Matrix([[particle_mass,0],
        [0,particle_mass],])
    for i in range(num_particles[None]):
        M[i,i] = m

@ti.kernel
# Jii=-k(I-r/|x_ij|(I-x_ij_norm*x_ij_norm^T))
def update_jacobi_matrix():
    I = ti.Matrix([[1.0, 0.0],
        [0.0, 1.0],])
    for i,d in J:
        J[i,d] *=0.0
        for j in range(num_particles[None]):
            if(rest_length[i,j] != 0) and (d == i or d == j):
                x_ij = x[i] - x[j]
                norm_x_ij = x_ij.norm()
                unit_x_ij = x_ij / norm_x_ij
                mat = unit_x_ij.outer_product(unit_x_ij)
                #Y = k * rest_length
                if d == i:
                    J[i,d] += -spring_Y[None] * (I / rest_length[i,j] - (I - mat) / norm_x_ij)
                else:
                    J[i,d] += spring_Y[None] * (I / rest_length[i,j] - (I - mat) / norm_x_ij)

               
@ti.kernel
def update_A_matrix():
    for i, j in A:
        A[i,j] = M[i,j] - dt * dt * J[i,j]


@ti.kernel
def update_F_vector():
    for i in range(num_particles[None]):
        F[i] = particle_mass * gravity

    for i,j in rest_length:
        if rest_length[i,j] != 0:
            x_ij = x[i] - x[j]
            d_ij = x_ij.normalized()
            F[i] += -spring_Y[None] * (x_ij.norm() / rest_length[i,j] - 1) * d_ij
             #dashpot damping: relative moving
            v_rel = (v[i] - v[j]).dot(d_ij)
            F[i]+=-dashpot_damping[None] * v_rel * d_ij
        else:
            pass

@ti.kernel
def update_b_vector():
    for i in range(num_particles[None]):
        v_star = v[i] * ti.exp(-dt * drag_damping[None])
        b[i] = M[i,i] @ v_star + dt * F[i]


@ti.func
def collide_with_walls(particle_id:ti.i32):
    for d in ti.static(range(2)):
        if x[particle_id][d] < 0:
            x[particle_id][d] = 0
            v[particle_id][d] = 0
        if x[particle_id][d] > 1:
            x[particle_id][d] = 1
            v[particle_id][d] = 0

@ti.kernel
def update_position():
    for i in range(num_particles[None]):
        
        #collide with four walls
        collide_with_walls(i)

        if not fixed[i]:
            x[i] += dt * v[i]
        else:
            v[i] = ti.Vector([0,0])
