
import taichi as ti
import numpy as np
import random
import os

write_to_disk = False
ti.init(arch=ti.gpu, debug=False)  # Try to run on GPU
quality = 1  # Use a larger value for higher-res simulations
n_grid = (40 * quality, 80 * quality)
n_particle_x = ti.field(dtype=ti.i32,shape=())
n_particle_y = ti.field(dtype=ti.i32,shape=())
n_elements = ti.field(dtype=ti.i32,shape=())
tetromino_size = 0.05
change_size = 3
gravity = ti.Vector([0.0,-9.8]) * 5.0

dx, inv_dx = 1 / n_grid[1], n_grid[1]
dt = 1e-4 / quality
p_vol, p_rho = (dx * 0.5) ** 2, 1
p_mass = p_vol * p_rho
E, nu = 0.15e4, 0.2  # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters
max_num_particles = 1024 * 16
dim = 2
x = ti.Vector.field(dim, dtype=float)  # position
v = ti.Vector.field(dim, dtype=float)  # velocity
C = ti.Matrix.field(dim, dim, dtype=float)  # affine velocity field
F = ti.Matrix.field(dim, dim, dtype=float)  # deformation gradient
material = ti.field(dtype=int)  # material id
Jp = ti.field(dtype=float)  # plastic deformation
B = ti.Matrix.field(dim,dim,dtype=float)
force = ti.Vector.field(dim,dtype=float)
vertices = ti.Vector.field(3, dtype=int) 
WD = ti.Matrix.field(dim,dim,dtype=float)
ti.root.dynamic(ti.i, max_num_particles).place(x, v, C, F, material, Jp, B, force, vertices, WD)
cur_num_particles = ti.field(ti.i32, shape=())

grid_v = ti.Vector.field(dim, dtype=float, shape=n_grid)
grid_m = ti.field(dtype=float, shape=n_grid)
cur_num_vertices = ti.field(ti.i32, shape=())
#====================================================================
def mesh_point(i,j,n_particle_y):
    return i * n_particle_y + j

@ti.func
def compute_T(vertices):
    a = x[vertices.x] - x[vertices.z] #a-c
    b = x[vertices.y] - x[vertices.z] #b-c
    return ti.Matrix([[a.x, b.x], [a.y, b.y]])

@ti.kernel
def compute_force(): 
    for e in vertices:
        F_tmp = compute_T(vertices[e]) @ B[e]
        J = F_tmp.determinant()
        mu, la = mu_0 * 1e-3, lambda_0 * 1e-3
        PK1 = mu * (F_tmp - F_tmp.transpose().inverse()) + la * ti.log(J) * F_tmp.transpose().inverse()
        H = PK1 @ WD[e] * 1e-2 * (8 * n_grid[1] ** 2)
        tmp_x = ti.Vector([H[0, 0], H[1, 0]])
        tmp_y = ti.Vector([H[0, 1], H[1, 1]])
        if (vertices[e] != [0,0,0]).all():
            force[vertices[e].x] += tmp_x
            force[vertices[e].y] += tmp_y
            force[vertices[e].z] += -tmp_x - tmp_y
#=====================================================================
@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0

    # p2g
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

        affine = p_mass * C[p]
        if material[p] != 3:
            F[p] = (ti.Matrix.identity(ti.f32, 2) + dt * C[p]) @ F[p]
            U, sig, V = ti.svd(F[p])
            #jelly
            if material[p] == 2:
                J = sig[0, 0] * sig[1, 1]
                h = 0.3
                mu, la = mu_0 * h, lambda_0 * h
                #corotated model PK1
                stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + la * (J - 1) * J * ti.Matrix.identity(ti.f32, 2)
                affine += -4 * dt * (inv_dx ** 2) * p_vol * stress
            #liquid
            elif material[p] == 0:
                J = sig[0, 0] * sig[1, 1]
                F[p] = ti.Matrix.identity(ti.f32, 2) * ti.sqrt(J)
                #mu=0 in corotated model(lazy solution)
                affine += -4 * dt * (inv_dx ** 2) * p_vol * E * (J - 1) * ti.Matrix.identity(ti.f32, 2)
            #snow
            elif material[p] == 1:
                #box yielding criterion
                h = max(0.1, min(5, ti.exp(10 * (1.0 - Jp[p])))) # Hardening coefficient: snow gets harder when compressed
                mu, la = mu_0 * h, lambda_0 * h
                J = 1.0
                for d in ti.static(range(2)):
                    new_sig = min(max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3) #elastoplastic
                    Jp[p] *= sig[d, d] / new_sig
                    sig[d, d] = new_sig
                    J *= new_sig
                F[p] = U @ sig @ V.transpose()
                # corotated model but different F[p]
                stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + la * (J - 1) * J * ti.Matrix.identity(ti.f32, 2)
                affine += -4 * dt * (inv_dx ** 2) * p_vol * stress

        for i, j in ti.static(ti.ndrange(3, 3)):  # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos + force[p] * p_mass)
            grid_m[base + offset] += weight * p_mass

    for i, j in grid_m:
        if grid_m[i, j] > 0:  # No need for epsilon here
            grid_v[i,j] = (1 / grid_m[i, j]) * grid_v[i,j]  # Momentum to velocity
            grid_v[i, j] += dt * gravity  # gravity
            if i < 3 and grid_v[i, j][0] < 0:grid_v[i, j][0] = 0  # Boundary conditions
            if i > n_grid[0] - 3 and grid_v[i, j][0] > 0: grid_v[i, j][0] = 0
            if j < 3 and grid_v[i, j][1] < 0: grid_v[i, j] = ti.Vector([0, 0])
            if j > n_grid[1] - 3 and grid_v[i, j][1] > 0: grid_v[i, j][1] = 0

    # g2p
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):  # loop over 3x3 grid node neighborhood
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]  # advection



#=====================================================================
num_per_tetromino_square = 128
num_per_tetromino = num_per_tetromino_square * 4
staging_tetromino_x = ti.Vector.field(dim, dtype=float, shape=num_per_tetromino)


class StagingTetromino(object):
    def __init__(self, x):

        self._x_field = x
        self.material_idx = 0

        self._offsets = np.array([[[0, -1], [1, 0], [0, -2]],
            [[1, 1], [-1, 0], [1, 0]],
            [[0, -1], [-1, 0], [0, -2]],
            [[0, 1], [1, 0], [1, -1]],
            [[1, 0], [2, 0], [-1, 0]],
            [[0, 1], [1, 1], [1, 0]],
            [[-1, 0], [1, 0], [0, 1]],])
        self._x_np_canonical = None
        self.left_width = 0
        self.right_width = 0
        self.lower_height = 0
        self.upper_height = 0

    def regenerate(self, mat, kind):
        self.material_idx = mat
        shape = (num_per_tetromino, dim)
        base = cur_num_particles[None]

        x = np.zeros(shape=shape, dtype=np.float32)
        if self.material_idx == 0 or self.material_idx == 1 or self.material_idx == 2:
            for i in range(1, 4):
                begin = i * num_per_tetromino_square
                x[begin:(begin + num_per_tetromino_square)] = self._offsets[kind, (i - 1)]
            x += np.random.rand(*shape)
            x *= tetromino_size

        elif self.material_idx == 3:
            n_particle_x[None], n_particle_y[None] = 10 * quality, 10 * quality 
            verticesbase = cur_num_vertices[None]
            base = cur_num_particles[None]
            #change tetromino size according to tetromino size, should built in
            #tetromino, but mesh is hard to build
            #n_particle_x[None] += (self._offsets[kind, 0][0] +
            #self._offsets[kind, 1][0] + self._offsets[kind, 2][0]) *
            #change_size
            #n_particle_y[None] += (self._offsets[kind, 0][1] +
            #self._offsets[kind, 1][1] + self._offsets[kind, 2][1]) *
            #change_size
            n_elements[None] = (n_particle_x[None] - 1) * (n_particle_y[None] - 1) * 2

            for j, k in ti.static(ti.ndrange(n_particle_x[None], n_particle_y[None])):
                p = j * n_particle_y[None] + k 
                x_coords, y_coords = j / 10.0, k / 10.0
                x[p] = ti.Vector([x_coords, y_coords])
            x *= tetromino_size * 2.0

            for j, k in ti.static(ti.ndrange(n_particle_x[None] - 1, n_particle_y[None] - 1)):
                e = j * (n_particle_y[None] - 1) + k
                p1, p2, p3, p4 = mesh_point(j, k,n_particle_y[None]) + base, mesh_point(j, k + 1,n_particle_y[None]) + base, mesh_point(j + 1, k,n_particle_y[None]) + base, mesh_point(j + 1, k + 1,n_particle_y[None]) + base
                vertices[e * 2 + verticesbase] = ti.Vector([p1, p3, p2])
                vertices[e * 2 + 1 + verticesbase] = ti.Vector([p2, p4, p3])

            cur_num_vertices[None] += n_elements[None]


        self._x_np_canonical = x
        self.left_width = tetromino_size * abs(min(self._offsets[kind, :][:, 0]))
        self.right_width = tetromino_size * abs(max(self._offsets[kind, :][:, 0]) + 1)
        self.lower_height = tetromino_size * abs(min(self._offsets[kind, :][:, 1]))
        self.upper_height = tetromino_size * abs(max(self._offsets[kind, :][:, 1]) + 1)
        

    def update_center(self, center):
        self._x_field.from_numpy(np.clip(self._x_np_canonical + center, 0, 1))

    def rotate(self, radius:int):
        theta = np.radians(radius)
        c, s = np.cos(theta), np.sin(theta)
        m = np.array([[c, -s], [s, c]], dtype=np.float32)
        x = m @ self._x_np_canonical.T
        self._x_np_canonical = x.T

        self.right_width, self.lower_height, self.left_width, self.upper_height = \
            self.lower_height, self.left_width, self.upper_height, self.right_width

    def compute_center(self, mouse, l_bound, r_bound):
        r = staging_tetromino.right_width
        l = staging_tetromino.left_width
        if mouse[0] + r > r_bound:
            x = r_bound - r
        elif mouse[0] - l < l_bound:
            x = l_bound + l
        else:
            x = mouse[0]
        return np.array([x, 0.8], dtype=np.float32)

staging_tetromino = StagingTetromino(staging_tetromino_x)


@ti.kernel
def drop_staging_tetromino(mat: int):
    base = cur_num_particles[None]
    verticesbase = cur_num_vertices[None] - n_elements[None]

    for i in staging_tetromino_x:
        bi = base + i
        material[bi] = mat
        if material[bi] == 0 or material[bi] == 1 or material[bi] == 2:
            x[bi] = staging_tetromino_x[i]
            v[bi] = ti.Matrix([0, -2])
            F[bi] = ti.Matrix([[1, 0], [0, 1]])
            Jp[bi] = 1
            cur_num_particles[None] += 1
        elif material[bi] == 3 and i < n_particle_x[None] * n_particle_y[None]:
            x[bi] = staging_tetromino_x[i]
            v[bi] = ti.Matrix([0, -2])
            C[bi] = ti.Matrix([[1, 0], [0, 1]])
            cur_num_particles[None] += 1

    for _ in range(1):
        if mat == 3:
            for i, j in ti.ndrange(n_particle_x[None] - 1, n_particle_y[None] - 1):
                #if print vertices here, next fem square will be eliminate
                e = i * (n_particle_y[None] - 1) + j
                tmp = compute_T(vertices[e * 2 + verticesbase])
                B[e * 2 + verticesbase] = tmp.inverse()
                WD[e * 2 + verticesbase] = -ti.abs(tmp.determinant()) / 2 * tmp.inverse().transpose()
                tmp = compute_T(vertices[e * 2 + 1 + verticesbase])
                B[e * 2 + 1 + verticesbase] = tmp.inverse()
                WD[e * 2 + 1 + verticesbase] = -ti.abs(tmp.determinant()) / 2 * tmp.inverse().transpose()




#============================================================
def main():
    os.makedirs('frames', exist_ok=True)
    gui = ti.GUI("MLS-MPM-",
                 res=(256, 512),
                 background_color=0x112F41)

    def gen_mat_and_kind():
        material_id = random.randint(0, 3)
        tetromino_kind = random.randint(0, 6)
        return material_id, tetromino_kind

    staging_tetromino.regenerate(*gen_mat_and_kind())

    last_action_frame = -1e10
    for f in range(100000):
        padding = 0.025
        segments = 20
        step = (1 - padding * 4) / (segments - 0.5) / 2
        for i in range(segments):
            gui.line(begin=(padding * 2 + step * 2 * i, 0.8),end=(padding * 2 + step * (2 * i + 1), 0.8),radius=1.5,color=0xFF8811)
        gui.line(begin=(padding * 2, padding),end=(1 - padding * 2, padding),radius=2)
        gui.line(begin=(padding * 2, 1 - padding),end=(1 - padding * 2, 1 - padding),radius=2)
        gui.line(begin=(padding * 2, padding),end=(padding * 2, 1 - padding),radius=2)
        gui.line(begin=(1 - padding * 2, padding),end=(1 - padding * 2, 1 - padding),radius=2)

        if gui.get_event(ti.GUI.PRESS):
            ev_key = gui.event.key
            if ev_key in [ti.GUI.ESCAPE, ti.GUI.EXIT]: break
            elif ev_key == 'e':
                staging_tetromino.rotate(90)
            elif ev_key == 'q':
                staging_tetromino.rotate(-90)
            elif ev_key == ti.GUI.SPACE:
                if cur_num_particles[None] + num_per_tetromino < max_num_particles:
                    drop_staging_tetromino(staging_tetromino.material_idx)
                print('# particles =', cur_num_particles[None])
                staging_tetromino.regenerate(*gen_mat_and_kind())
                last_action_frame = f
        mouse = gui.get_cursor_pos()
        mouse = (mouse[0] * 0.5, mouse[1])

        right_bound = 0.5 - padding
        left_bound = padding
        
        staging_tetromino.update_center(staging_tetromino.compute_center(mouse, left_bound, right_bound))

        for s in range(int(2e-3 // dt)):
            force.fill(0)
            compute_force()
            substep()

        colors = np.array([0xA6B5F7, 0xEEEEF0, 0xED553B, 0xFE2E44, 0x6D35CB, 0xEDE53B, 0x26A5A7, 0xEDE53B], dtype=np.uint32)
        particle_radius = 2.3
        gui.circles(x.to_numpy() * [[2, 1]],
                    radius=particle_radius,
                    color=colors[material.to_numpy()])

        if last_action_frame + 40 < f:
            gui.circles(staging_tetromino_x.to_numpy() * [[2, 1]],
                        radius=particle_radius,
                        color=int(colors[staging_tetromino.material_idx]))

            if staging_tetromino.material_idx == 0:
                mat_text = 'Liquid'
            elif staging_tetromino.material_idx == 1:
                mat_text = 'Snow'
            elif staging_tetromino.material_idx == 2:
                mat_text = 'Jelly'
            elif staging_tetromino.material_idx == 3:
                mat_text = 'Fem'
            #gui.text(mat_text, (0.42, 0.97), font_size=30, color=colors[staging_tetromino.material_idx])

        #gui.text('Taichi Tetris', (0.07, 0.97), font_size=20)

        
        if write_to_disk:
            gui.show(f'frames/{f:05d}.png')
        else:
            gui.show()


if __name__ == '__main__':
    main()