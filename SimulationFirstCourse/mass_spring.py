import taichi as ti
import numpy as np
from mass_spring_implicit import *


@ti.kernel
def add_particle(pos_x:ti.f32,pos_y:ti.f32,fixed_:ti.i32):
    new_particle_id = num_particles[None]

    x[new_particle_id] = ti.Vector([pos_x,pos_y])
    fixed[new_particle_id] = fixed_

    for i in range(num_particles[None]):
        if(x[new_particle_id] - x[i]).norm() < 0.15:
            rest_length[i,new_particle_id] = 0.1
            rest_length[new_particle_id,i] = 0.1

    num_particles[None]+=1


# euler FDM explicit method
#================================================================
@ti.kernel
def euler_explicit_substep():
    n = num_particles[None]

    #compute force
    for i in range(n):
        #gravity
        f[i] = gravity * particle_mass
        for j in range(n):
            if rest_length[i,j] != 0:#has spring between ij
                x_ij = x[i] - x[j]
                d_ij = x_ij.normalized()
                #spring force
                f[i]+=-spring_Y[None] * (x_ij.norm() / rest_length[i,j] - 1) * d_ij
                #dashpot damping: relative moving
                v_rel = (v[i] - v[j]).dot(d_ij)
                f[i]+=-dashpot_damping[None] * v_rel * d_ij
    
    for i in range(n):
        if not fixed[i]:
            v[i]+=dt * f[i] / particle_mass
            v[i]*=ti.exp(-dt * drag_damping[None])#drag damping
            x[i]+=dt * v[i]
        else:
            v[i] = ti.Vector([0,0])

        #collide with four walls
        collide_with_walls(i)
#================================================================


#runge kutta explicit method
#================================================================
v2 = ti.Vector.field(2, dtype = ti.f32, shape = max_num_particles)
v3 = ti.Vector.field(2, dtype = ti.f32, shape = max_num_particles)
v4 = ti.Vector.field(2, dtype = ti.f32, shape = max_num_particles)

f1 = ti.Vector.field(2, dtype = ti.f32, shape = max_num_particles)
f2 = ti.Vector.field(2, dtype = ti.f32, shape = max_num_particles)
f3 = ti.Vector.field(2, dtype = ti.f32, shape = max_num_particles)
f4 = ti.Vector.field(2, dtype = ti.f32, shape = max_num_particles)
    
@ti.kernel
def RK4_explicit_substep():
    # compute force and new velocity
    n = num_particles[None]

    #initialize velocity
    for i in range(n):
        v[i] *= ti.exp(-dt * drag_damping[None]) # damping

    #1
    for i in range(n):
        f1[i] = gravity * particle_mass
        for j in range(n):
            if rest_length[i, j] != 0:
                x_ij = x[i] - x[j]
                d_ij = x_ij.normalized()
                f1[i] += -spring_Y[None] * (x_ij.norm() / rest_length[i,j] - 1) * d_ij
                v_rel = (v[i] - v[j]).dot(d_ij)
                f1[i] += -dashpot_damping[None] * v_rel * d_ij

    #2
    for i in range(n):
        v2[i] = v[i] + (dt / 2) * (f1[i] / particle_mass)
    for i in range(n):
        f2[i] = gravity * particle_mass
        for j in range(n):
            if rest_length[i, j] != 0:
                x_ij = x[i] - x[j] + (dt / 2) * (v[i] - v[j])
                d_ij = x_ij.normalized()
                f2[i] += -spring_Y[None] * (x_ij.norm() / rest_length[i,j] - 1) * d_ij
                v_rel = (v[i] - v[j]).dot(d_ij)
                f2[i] += -dashpot_damping[None] * v_rel * d_ij

    #3
    for i in range(n):
        v3[i] = v[i] + (dt / 2) * (f2[i] / particle_mass)
    for i in range(n):
        f3[i] = gravity * particle_mass
        for j in range(n):
            if rest_length[i, j] != 0:
                x_ij = x[i] - x[j] + (dt / 2) * (v2[i] - v2[j])
                d_ij = x_ij.normalized()
                f3[i] += -spring_Y[None] * (x_ij.norm() / rest_length[i,j] - 1) * d_ij
                v_rel = (v[i] - v[j]).dot(d_ij)
                f3[i] += -dashpot_damping[None] * v_rel * d_ij

    #4
    for i in range(n):
        v4[i] = v[i] + dt * (f3[i] / particle_mass)
    for i in range(n):
        f4[i] = gravity * particle_mass
        for j in range(n):
            if rest_length[i, j] != 0:
                x_ij = x[i] - x[j] + dt * (v3[i] - v3[j])
                d_ij = x_ij.normalized()
                f4[i] += -spring_Y[None] * (x_ij.norm() / rest_length[i,j] - 1) * d_ij
                v_rel = (v[i] - v[j]).dot(d_ij)
                f4[i]+=-dashpot_damping[None] * v_rel * d_ij
        

    for i in range(n):
        if not fixed[i]:
            x[i] += (dt / 6) * (v[i] + 2 * v2[i] + 2 * v3[i] + v4[i])
            v[i] += (dt / 6) * (f1[i] + 2 * f2[i] + 2 * f3[i] + f4[i]) / particle_mass
            v[i] *= ti.exp(-dt * drag_damping[None])#drag damping
        else:
            v[i] = ti.Vector([0,0])

        collide_with_walls(i)
        
#================================================================


# conjugate gradient method
#================================================================
d = ti.Vector.field(2, dtype = ti.f32, shape = max_num_particles)
r = ti.Vector.field(2, dtype = ti.f32, shape = max_num_particles)
a = ti.field(dtype = ti.f32, shape = max_num_particles)

@ti.kernel
def conjugate_gradients_init():
    for i in range(num_particles[None]):
        Ax0_i = ti.Vector([0.0, 0.0])
        for j in range(num_particles[None]):
            Ax0_i += A[i,j] @ v[j]
        d[i] = b[i] - Ax0_i
        r[i] = b[i] - Ax0_i

@ti.kernel
def conjugate_gradients_iteration():
    for i in range(num_particles[None]):
        a[i] = r[i].dot(r[i])
        dTA_i = ti.Vector([0.0, 0.0])
        for j in range(num_particles[None]):
            dTA_i[0] += d[i][0] * A[j, i][0, 0] + d[i][1] * A[j, i][1, 0]
            dTA_i[1] += d[i][1] * A[j, i][0, 1] + d[i][1] * A[j, i][1, 1]
        a[i] /= dTA_i.dot(d[i])

    for i in range(num_particles[None]):
        v[i] += a[i] * d[i]

    for i in range(num_particles[None]):
        Ad_i = ti.Vector([0.0, 0.0])
        for j in range(num_particles[None]):
            Ad_i += A[i, j] @ d[j]

        rTr = r[i].dot(r[i])
        r[i] -= a[i] * Ad_i
        beta = r[i].dot(r[i]) / rTr
        d[i] *= beta
        d[i] += r[i]


def conjugate_gradients_substep(itertimes:ti.i32):
    update_mass_matrix()
    update_jacobi_matrix()
    update_A_matrix()
    update_F_vector()
    update_b_vector()

    conjugate_gradients_init()
    for step in range(itertimes):
        conjugate_gradients_iteration()
    update_position()

#================================================================

#jacobi_iterative_method
#================================================================
@ti.kernel
def jacobi_iteration():
    for i in range(num_particles[None]):
        r = b[i]
        for j in range(num_particles[None]):
            if i != j:
                r -= A[i, j] @ v[j]
        new_v[i] = A[i, i].inverse() @ r

    for i in range(num_particles[None]):
        v[i] = new_v[i]

def jacobi_implicit_substep(itertimes:ti.i32):
    update_mass_matrix()
    update_jacobi_matrix()
    update_A_matrix()
    update_F_vector()
    update_b_vector()

    for step in range(itertimes):
        jacobi_iteration()
    update_position()
#================================================================

#gauss seidel
#================================================================
inv_L = ti.Matrix.field(2, 2, dtype = ti.f32, shape = (max_num_particles, max_num_particles))
invL_v = ti.Vector.field(2, dtype = ti.f32, shape = (max_num_particles))

@ti.kernel
def gauss_seidel_init():
    O = ti.Matrix([[0.0, 0.0],
        [0.0, 0.0]])
    I = ti.Matrix([[1.0, 0.0],
        [0.0, 1.0]])

    for i, j in inv_L:
        inv_L[i, j] = O
    for i in range(num_particles[None]):
        inv_L[i, i] = I

    # solve L-1, inv of A's upper triangular matrix
    for i in range(num_particles[None]):
        inv_ii = A[i, i].inverse()
        for j in range(i):
            n = A[j, i] @ inv_ii
            inv_L[j, i] -= n

    for i in range(num_particles[None]):
        inv_ii = A[i, i].inverse()
        for j in range(i, num_particles[None]):
            inv_L[i, j] = inv_L[i, j] @ inv_ii

# D = diag(A), L = -tril(A), U = -triu(A)
# (D - L)x(i+1) = Ux(i) + b
# x(i+1) = (D - L)^-1 * (Ux(i) + b)
@ti.kernel
def gauss_seidel_iteration():
    # solve Ux(i) + b
    for i in range(num_particles[None]):
        new_v[i] = b[i]
        for j in range(i):
            new_v[i] -= A[i, j] @ v[j]

    # solve (D - L)^-1 * (Ux(i) + b)
    for i in range(num_particles[None]):
        invL_v[i] = ti.Vector([0.0, 0.0])
        for j in range(i, num_particles[None]):
            invL_v[i] += inv_L[i, j] @ new_v[j]

    # update v
    for i in range(num_particles[None]):
        v[i] = invL_v[i]

def gauss_seidel_substep(itertimes:ti.i32):
    update_mass_matrix()
    update_jacobi_matrix()
    update_A_matrix()
    update_F_vector()
    update_b_vector()

    gauss_seidel_init()
    for step in range(itertimes):
        gauss_seidel_iteration()
    update_position()
#================================================================

@ ti.kernel
def attract(pos_x:ti.f32,pos_y:ti.f32):
    for i in range(num_particles[None]):
        v[i] += -dt * substeps * (x[i] - ti.Vector([pos_x,pos_y])) * attract_force


def init_pos():
    N = 4
    for i,j in ti.ndrange(N + 1, N + 1):
        k = i * (N + 1) + j
        pos = ti.Vector([i,j]) / N * 0.3 + ti.Vector([0.25,0.25])
        if (i == N and j == N) or (i==0 and j==N):
            add_particle(pos.x,pos.y, 1)
        else:
            add_particle(pos.x,pos.y, 0)

def calculate_mode(mode:ti.i32):
    if mode == 0:
        euler_explicit_substep()
    elif mode == 1:
        RK4_explicit_substep()
    elif mode == 2:
        jacobi_implicit_substep(10)
    elif mode == 3:
        gauss_seidel_substep(10)
    elif mode == 4:
        conjugate_gradients_substep(1)


def main():
    gui = ti.GUI("spring system", res=(512,512),background_color=0x112F41)

    spring_Y[None] = 1000
    drag_damping[None] = 1
    dashpot_damping[None] = 50
    mode = 0

    

    while True:
               
        calculateSolver = ['Euler Explicit','RK4 Explicit','Jacobi Implicit','Gauss Seidel','Conjugate Gradients']
        colors = np.array([0xA6B5F7, 0xEEEEF0, 0xED553B, 0xFE2E44, 0x6D35CB], dtype=np.uint32)
                     
        if gui.is_pressed(ti.GUI.RMB):
            c = gui.get_cursor_pos()
            attract(c[0],c[1])


        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.LMB:
                add_particle(e.pos[0], e.pos[1], int(gui.is_pressed(ti.GUI.SHIFT)))
            elif e.key == 's': 
                if gui.is_pressed('Shift'):
                    spring_Y[None] -= 50
                else:
                    spring_Y[None] += 50
            elif e.key == 'd':
                if gui.is_pressed('Shift'):
                    dashpot_damping[None] -= 10
                else:
                    dashpot_damping[None] += 10
            elif e.key == '1': mode = 0
            elif e.key == '2': mode = 1
            elif e.key == '3': mode = 2
            elif e.key == '4': mode = 3
            elif e.key == '5': mode = 4
            elif e.key == 'c': 
                #show cloth
                num_particles[None] = 0
                rest_length.fill(0)
                init_pos()
            elif e.key == 'r':
                num_particles[None] = 0
                rest_length.fill(0)

        #gui.text(calculateSolver[mode], (0.37, 0.97), font_size=30, color=colors[mode])
        #gui.text(content = f'Spring stiffness {spring_Y[None]:.1f}', font_size = 15, pos = (0, 0.95), color = 0xdddddd)
        #gui.text(content = f'Dashpot damping {dashpot_damping[None]:.1f}', font_size = 15, pos = (0, 0.90), color = 0xdddddd)

        for step in range(substeps):
            calculate_mode(mode)



        X = x.to_numpy()#reads all particle positions to a numpy array
        for i in range(num_particles[None]):
            for j in range(num_particles[None]):
                if rest_length[i,j] != 0:
                    gui.line(begin=X[i],end=X[j],color=0x88888,radius=2)
        for i in range(num_particles[None]):
            c = 0x0 if fixed[i] else 0xFF0000
            gui.circle(X[i],color=c,radius=5)


        #main loop
        gui.show()

if __name__ == '__main__':
    main()
