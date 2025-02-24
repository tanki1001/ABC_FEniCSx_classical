import os
import numpy as np
from scipy import special
import scipy.linalg as la
from scipy.sparse import csr_matrix 
from scipy.io import savemat
from sympy import symbols, diff, lambdify
import sympy as sy
import matplotlib.pyplot as plt

import pyvista

from tqdm import tqdm
from time import time

from abc import ABC, abstractmethod

import gmsh
from dolfinx import plot
from basix.ufl import element
from dolfinx.io import gmshio
import dolfinx.mesh as msh
from mpi4py import MPI
from dolfinx.fem import Function, functionspace, assemble, form, petsc, Constant, assemble_scalar
from dolfinx.fem.petsc import LinearProblem
from ufl import (TestFunction, TrialFunction, TrialFunctions,
                 dx, grad, inner, Measure, variable, FacetNormal, CellNormal)
from petsc4py import PETSc
import slepc4py.SLEPc as SLEPc

from geometries import cubic_domain, spherical_domain, half_cubic_domain, broken_cubic_domain
from operators_POO import (Mesh, Loading, plot_analytical_result_sigma)
# conda activate env3-10-complex    
# Choice of the geometry among provided ones
geometry1 = 'cubic'
geometry2 = 'small'
geometry  = geometry1 + '_'+ geometry2

if   geometry2 == 'small':
    side_box = 0.11
    lc       = 8e-3
elif geometry2 == 'large':
    side_box = 0.40
    lc       = 2e-2
else :
    print("Enter your own side_box and mesh size in the code")
    side_box = 0.40
    lc       = 1e-2 #Typical mesh size : Small case : 8e-3 Large case : 2e-3

if   geometry1 == 'cubic':
    geo_fct = cubic_domain
elif geometry1 == 'spherical':
    geo_fct = spherical_domain
elif geometry1 == 'half_cubic':
    geo_fct = half_cubic_domain
elif geometry1 == 'broken_cubic':
    geo_fct = broken_cubic_domain
else :
    print("WARNING : May you choose an implemented geometry")

def save_results(file_str, z_center):
    with open('no_Q_results/' + file_str, 'w') as fichier:    
        for i in range(len(freqvec)):        
            fichier.write('{}\t{}\n'.format(freqvec[i], z_center[i]))

# Simulation parameters
radius  = 0.1                               # Radius of the baffle
rho0    = 1.21                              # Density of the air
c0      = 343.8                             # Speed of sound in air
freqvec = np.arange(80, 2001, 20)           # List of the frequencies

dimP = 2
dimQ = dimP


mesh_    = Mesh(dimP, dimQ, side_box, radius, lc, geo_fct)

dx, ds, dx1 = mesh_.integral_mesure()
P, Q = mesh_.fonction_spaces()
xref = mesh_.xref
mesh = mesh_.mesh
submesh = mesh_.submesh
entity_maps_mesh = mesh_.entity_maps_mesh

p, q = TrialFunction(P), TrialFunction(Q)
v, u = TestFunction(P), TestFunction(Q)

k = inner(grad(p), grad(v)) * dx
m = inner(p, v) * dx
c = inner(q, v) * ds(3)

fx1_p = Function(P)
fx1_p.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))

fx1_q = Function(Q)
fx1_q.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))

n   = FacetNormal(mesh)
dp  = inner(grad(p), n) # dp/dn = grad(p) * n
ddp = inner(grad(dp), n) # d^2p/dn^2 = grad(dp/dn) * n = grad(grad(p) * n) * n
g1  = inner((1-1)*ddp, u)*ds(3)
g2  = inner(p, u)*ds(3)
g3  = inner(4*fx1_p*p, u)*ds(3)
g4  = inner(2*fx1_p**2*p, u)*ds(3)

e1  = inner(4*fx1_q*q, u)*dx1
e2  = inner(2*q, u)*dx1

K = form(k)
Ks = petsc.assemble_matrix(K)
print(f"Size of the stiffness matrix: {Ks.getSize()}")
M = form(m)
C = form(-c, entity_maps=entity_maps_mesh)
Cs = petsc.assemble_matrix(C)
print(f"Size of the damping matrix: {Cs.getSize()}")


G2 = form(g2, entity_maps=entity_maps_mesh)
G3 = form(g3, entity_maps=entity_maps_mesh)
G4 = form(g4, entity_maps=entity_maps_mesh)
G1_4 = form(g1 + g4, entity_maps=entity_maps_mesh)
G1_4s = petsc.assemble_matrix(G1_4)
print(f"Size of the G1_4 matrix: {G1_4s.getSize()}")

E1 = form(e1)
E2 = form(e2)


D1 = [[K,     C],
      [G1_4, E1]]
D1 = petsc.assemble_matrix_block(D1)
D1.assemble()
print(f"Size of the D1 matrix: {D1.getSize()}")

D2_00 = form(inner(Constant(mesh, PETSc.ScalarType(0))*p, v) * dx)
D2_01 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, v) * ds(3), entity_maps=entity_maps_mesh)
D2 = [[D2_00,   D2_01],
      [G3,         E2]]
D2 = petsc.assemble_matrix_block(D2)
D2.assemble()

D3_01 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, v) * ds(3), entity_maps=entity_maps_mesh)
D3_11 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, u) * dx1)
D3 = [[M,  D3_01],
      [G2, D3_11]]
D3 = petsc.assemble_matrix_block(D3)
D3.assemble()

f = form(inner(1, v) * ds(1))
zero = form(inner(Constant(submesh, PETSc.ScalarType(0)), u) * dx1)

F = [f, zero]
F = petsc.assemble_vector_nest(F)

Pav1         = np.zeros(freqvec.size, dtype=np.complex128)
Psol1, Qsol1 = Function(P), Function(Q)
offset       = P.dofmap.index_map.size_local * P.dofmap.index_map_bs # This is a bit obscur so far

for ii in tqdm(range(freqvec.size)):
    freq = freqvec[ii]
    omega0 = 2*np.pi*freq
    k0 = omega0/c0

    Z = 0 + D1 + 1j*k0*D2 - k0**2*D3
    #Z.assemble() # <- I'm not sure that's necessary | To test : try to do the simulation with and without to see if it leads to the same results
    if ii == 0:
        print(f"Size of the global matrix: {Z.getSize()}")

    # Solve
    # Having the solver inside the loop or outside doesn't change the CPU time
    ksp = PETSc.KSP().create()
    ksp.setOperators(Z)
    ksp.setType("gmres") # Solver type 
    ksp.getPC().setType("lu") # Preconditionner type
    ksp.getPC().setFactorSolverType("mumps") # Various type of previous objects are available, and different tests have to be performed to find the best. Normaly this configuration provides best results

    X = F.copy()
    ksp.solve(F, X) # Inversion of the matrix

    Psol1.x.array[:offset] = X.array_r[:offset]
    Qsol1.x.array[:(len(X.array_r) - offset)] = X.array_r[offset:]

    Pav1[ii] = assemble_scalar(form(Psol1*ds(1)))
    ksp.destroy()
    X.destroy()
    Z.destroy()



surfarea = assemble_scalar(form(1*ds(1)))
k_output = 2*np.pi*freqvec/c0
Z_center = 1j*k_output* Pav1 / surfarea
#save_results(file_str, Z_center)

fig, ax = plt.subplots(figsize=(16,9))
plot_analytical_result_sigma(ax, freqvec, radius)
ax.plot(freqvec, Z_center.real, label = "alpha = 0 dim P = 2")



dimP = 4
dimQ = dimP


mesh_    = Mesh(dimP, dimQ, side_box, radius, lc, geo_fct)

dx, ds, dx1 = mesh_.integral_mesure()
P, Q = mesh_.fonction_spaces()
xref = mesh_.xref
mesh = mesh_.mesh
submesh = mesh_.submesh
entity_maps_mesh = mesh_.entity_maps_mesh

p, q = TrialFunction(P), TrialFunction(Q)
v, u = TestFunction(P), TestFunction(Q)

k = inner(grad(p), grad(v)) * dx
m = inner(p, v) * dx
c = inner(q, v) * ds(3)

fx1_p = Function(P)
fx1_p.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))

fx1_q = Function(Q)
fx1_q.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))

n   = FacetNormal(mesh)
dp  = inner(grad(p), n) # dp/dn = grad(p) * n
ddp = inner(grad(dp), n) # d^2p/dn^2 = grad(dp/dn) * n = grad(grad(p) * n) * n
g1  = inner((1-1)*ddp, u)*ds(3)
g2  = inner(p, u)*ds(3)
g3  = inner(4*fx1_p*p, u)*ds(3)
g4  = inner(2*fx1_p**2*p, u)*ds(3)

e1  = inner(4*fx1_q*q, u)*dx1
e2  = inner(2*q, u)*dx1

K = form(k)
Ks = petsc.assemble_matrix(K)
print(f"Size of the stiffness matrix: {Ks.getSize()}")
M = form(m)
C = form(-c, entity_maps=entity_maps_mesh)
Cs = petsc.assemble_matrix(C)
print(f"Size of the damping matrix: {Cs.getSize()}")


G2 = form(g2, entity_maps=entity_maps_mesh)
G3 = form(g3, entity_maps=entity_maps_mesh)
G4 = form(g4, entity_maps=entity_maps_mesh)
G1_4 = form(g1 + g4, entity_maps=entity_maps_mesh)
G1_4s = petsc.assemble_matrix(G1_4)
print(f"Size of the G1_4 matrix: {G1_4s.getSize()}")

E1 = form(e1)
E2 = form(e2)


D1 = [[K,     C],
      [G1_4, E1]]
D1 = petsc.assemble_matrix_block(D1)
D1.assemble()
print(f"Size of the D1 matrix: {D1.getSize()}")

D2_00 = form(inner(Constant(mesh, PETSc.ScalarType(0))*p, v) * dx)
D2_01 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, v) * ds(3), entity_maps=entity_maps_mesh)
D2 = [[D2_00,   D2_01],
      [G3,         E2]]
D2 = petsc.assemble_matrix_block(D2)
D2.assemble()

D3_01 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, v) * ds(3), entity_maps=entity_maps_mesh)
D3_11 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, u) * dx1)
D3 = [[M,  D3_01],
      [G2, D3_11]]
D3 = petsc.assemble_matrix_block(D3)
D3.assemble()

f = form(inner(1, v) * ds(1))
zero = form(inner(Constant(submesh, PETSc.ScalarType(0)), u) * dx1)

F = [f, zero]
F = petsc.assemble_vector_nest(F)

Pav1         = np.zeros(freqvec.size, dtype=np.complex128)
Psol1, Qsol1 = Function(P), Function(Q)
offset       = P.dofmap.index_map.size_local * P.dofmap.index_map_bs # This is a bit obscur so far

for ii in tqdm(range(freqvec.size)):
    freq = freqvec[ii]
    omega0 = 2*np.pi*freq
    k0 = omega0/c0

    Z = 0 + D1 + 1j*k0*D2 - k0**2*D3
    #Z.assemble() # <- I'm not sure that's necessary | To test : try to do the simulation with and without to see if it leads to the same results
    if ii == 0:
        print(f"Size of the global matrix: {Z.getSize()}")

    # Solve
    # Having the solver inside the loop or outside doesn't change the CPU time
    ksp = PETSc.KSP().create()
    ksp.setOperators(Z)
    ksp.setType("gmres") # Solver type 
    ksp.getPC().setType("lu") # Preconditionner type
    ksp.getPC().setFactorSolverType("mumps") # Various type of previous objects are available, and different tests have to be performed to find the best. Normaly this configuration provides best results

    X = F.copy()
    ksp.solve(F, X) # Inversion of the matrix

    Psol1.x.array[:offset] = X.array_r[:offset]
    Qsol1.x.array[:(len(X.array_r) - offset)] = X.array_r[offset:]

    Pav1[ii] = assemble_scalar(form(Psol1*ds(1)))
    ksp.destroy()
    X.destroy()
    Z.destroy()



surfarea = assemble_scalar(form(1*ds(1)))
k_output = 2*np.pi*freqvec/c0
Z_center = 1j*k_output* Pav1 / surfarea
#save_results(file_str, Z_center)


ax.plot(freqvec, Z_center.real, label = "alpha = 0 dim P = 4")


dimP = 2
dimQ = dimP


mesh_    = Mesh(dimP, dimQ, side_box, radius, lc, geo_fct)

dx, ds, dx1 = mesh_.integral_mesure()
P, Q = mesh_.fonction_spaces()
xref = mesh_.xref
mesh = mesh_.mesh
submesh = mesh_.submesh
entity_maps_mesh = mesh_.entity_maps_mesh

p, q = TrialFunction(P), TrialFunction(Q)
v, u = TestFunction(P), TestFunction(Q)

k = inner(grad(p), grad(v)) * dx
m = inner(p, v) * dx
c = inner(q, v) * ds(3)

fx1_p = Function(P)
fx1_p.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))

fx1_q = Function(Q)
fx1_q.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))

n   = FacetNormal(mesh)
dp  = inner(grad(p), n) # dp/dn = grad(p) * n
ddp = inner(grad(dp), n) # d^2p/dn^2 = grad(dp/dn) * n = grad(grad(p) * n) * n
g1  = inner(ddp, u)*ds(3)
g2  = inner(p, u)*ds(3)
g3  = inner(4*fx1_p*p, u)*ds(3)
g4  = inner(2*fx1_p**2*p, u)*ds(3)

e1  = inner(4*fx1_q*q, u)*dx1
e2  = inner(2*q, u)*dx1

K = form(k)
Ks = petsc.assemble_matrix(K)
print(f"Size of the stiffness matrix: {Ks.getSize()}")
M = form(m)
C = form(-c, entity_maps=entity_maps_mesh)
Cs = petsc.assemble_matrix(C)
print(f"Size of the damping matrix: {Cs.getSize()}")


G2 = form(g2, entity_maps=entity_maps_mesh)
G3 = form(g3, entity_maps=entity_maps_mesh)
G4 = form(g4, entity_maps=entity_maps_mesh)
G1_4 = form(g1 + g4, entity_maps=entity_maps_mesh)
G1_4s = petsc.assemble_matrix(G1_4)
print(f"Size of the G1_4 matrix: {G1_4s.getSize()}")

E1 = form(e1)
E2 = form(e2)


D1 = [[K,     C],
      [G1_4, E1]]
D1 = petsc.assemble_matrix_block(D1)
D1.assemble()
print(f"Size of the D1 matrix: {D1.getSize()}")

D2_00 = form(inner(Constant(mesh, PETSc.ScalarType(0))*p, v) * dx)
D2_01 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, v) * ds(3), entity_maps=entity_maps_mesh)
D2 = [[D2_00,   D2_01],
      [G3,         E2]]
D2 = petsc.assemble_matrix_block(D2)
D2.assemble()

D3_01 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, v) * ds(3), entity_maps=entity_maps_mesh)
D3_11 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, u) * dx1)
D3 = [[M,  D3_01],
      [G2, D3_11]]
D3 = petsc.assemble_matrix_block(D3)
D3.assemble()

f = form(inner(1, v) * ds(1))
zero = form(inner(Constant(submesh, PETSc.ScalarType(0)), u) * dx1)

F = [f, zero]
F = petsc.assemble_vector_nest(F)

Pav1         = np.zeros(freqvec.size, dtype=np.complex128)
Psol1, Qsol1 = Function(P), Function(Q)
offset       = P.dofmap.index_map.size_local * P.dofmap.index_map_bs # This is a bit obscur so far

for ii in tqdm(range(freqvec.size)):
    freq = freqvec[ii]
    omega0 = 2*np.pi*freq
    k0 = omega0/c0

    Z = 0 + D1 + 1j*k0*D2 - k0**2*D3
    #Z.assemble() # <- I'm not sure that's necessary | To test : try to do the simulation with and without to see if it leads to the same results
    if ii == 0:
        print(f"Size of the global matrix: {Z.getSize()}")

    # Solve
    # Having the solver inside the loop or outside doesn't change the CPU time
    ksp = PETSc.KSP().create()
    ksp.setOperators(Z)
    ksp.setType("gmres") # Solver type 
    ksp.getPC().setType("lu") # Preconditionner type
    ksp.getPC().setFactorSolverType("mumps") # Various type of previous objects are available, and different tests have to be performed to find the best. Normaly this configuration provides best results

    X = F.copy()
    ksp.solve(F, X) # Inversion of the matrix

    Psol1.x.array[:offset] = X.array_r[:offset]
    Qsol1.x.array[:(len(X.array_r) - offset)] = X.array_r[offset:]

    Pav1[ii] = assemble_scalar(form(Psol1*ds(1)))
    ksp.destroy()
    X.destroy()
    Z.destroy()



surfarea = assemble_scalar(form(1*ds(1)))
k_output = 2*np.pi*freqvec/c0
Z_center = 1j*k_output* Pav1 / surfarea
#save_results(file_str, Z_center)

ax.plot(freqvec, Z_center.real, label = "alpha = 1 dim P = 2")

dimP = 4
dimQ = dimP


mesh_    = Mesh(dimP, dimQ, side_box, radius, lc, geo_fct)

dx, ds, dx1 = mesh_.integral_mesure()
P, Q = mesh_.fonction_spaces()
xref = mesh_.xref
mesh = mesh_.mesh
submesh = mesh_.submesh
entity_maps_mesh = mesh_.entity_maps_mesh

p, q = TrialFunction(P), TrialFunction(Q)
v, u = TestFunction(P), TestFunction(Q)

k = inner(grad(p), grad(v)) * dx
m = inner(p, v) * dx
c = inner(q, v) * ds(3)

fx1_p = Function(P)
fx1_p.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))

fx1_q = Function(Q)
fx1_q.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))

n   = FacetNormal(mesh)
dp  = inner(grad(p), n) # dp/dn = grad(p) * n
ddp = inner(grad(dp), n) # d^2p/dn^2 = grad(dp/dn) * n = grad(grad(p) * n) * n
g1  = inner(ddp, u)*ds(3)
g2  = inner(p, u)*ds(3)
g3  = inner(4*fx1_p*p, u)*ds(3)
g4  = inner(2*fx1_p**2*p, u)*ds(3)

e1  = inner(4*fx1_q*q, u)*dx1
e2  = inner(2*q, u)*dx1

K = form(k)
Ks = petsc.assemble_matrix(K)
print(f"Size of the stiffness matrix: {Ks.getSize()}")
M = form(m)
C = form(-c, entity_maps=entity_maps_mesh)
Cs = petsc.assemble_matrix(C)
print(f"Size of the damping matrix: {Cs.getSize()}")


G2 = form(g2, entity_maps=entity_maps_mesh)
G3 = form(g3, entity_maps=entity_maps_mesh)
G4 = form(g4, entity_maps=entity_maps_mesh)
G1_4 = form(g1 + g4, entity_maps=entity_maps_mesh)
G1_4s = petsc.assemble_matrix(G1_4)
print(f"Size of the G1_4 matrix: {G1_4s.getSize()}")

E1 = form(e1)
E2 = form(e2)


D1 = [[K,     C],
      [G1_4, E1]]
D1 = petsc.assemble_matrix_block(D1)
D1.assemble()
print(f"Size of the D1 matrix: {D1.getSize()}")

D2_00 = form(inner(Constant(mesh, PETSc.ScalarType(0))*p, v) * dx)
D2_01 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, v) * ds(3), entity_maps=entity_maps_mesh)
D2 = [[D2_00,   D2_01],
      [G3,         E2]]
D2 = petsc.assemble_matrix_block(D2)
D2.assemble()

D3_01 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, v) * ds(3), entity_maps=entity_maps_mesh)
D3_11 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, u) * dx1)
D3 = [[M,  D3_01],
      [G2, D3_11]]
D3 = petsc.assemble_matrix_block(D3)
D3.assemble()

f = form(inner(1, v) * ds(1))
zero = form(inner(Constant(submesh, PETSc.ScalarType(0)), u) * dx1)

F = [f, zero]
F = petsc.assemble_vector_nest(F)

Pav1         = np.zeros(freqvec.size, dtype=np.complex128)
Psol1, Qsol1 = Function(P), Function(Q)
offset       = P.dofmap.index_map.size_local * P.dofmap.index_map_bs # This is a bit obscur so far

for ii in tqdm(range(freqvec.size)):
    freq = freqvec[ii]
    omega0 = 2*np.pi*freq
    k0 = omega0/c0

    Z = 0 + D1 + 1j*k0*D2 - k0**2*D3
    #Z.assemble() # <- I'm not sure that's necessary | To test : try to do the simulation with and without to see if it leads to the same results
    if ii == 0:
        print(f"Size of the global matrix: {Z.getSize()}")

    # Solve
    # Having the solver inside the loop or outside doesn't change the CPU time
    ksp = PETSc.KSP().create()
    ksp.setOperators(Z)
    ksp.setType("gmres") # Solver type 
    ksp.getPC().setType("lu") # Preconditionner type
    ksp.getPC().setFactorSolverType("mumps") # Various type of previous objects are available, and different tests have to be performed to find the best. Normaly this configuration provides best results

    X = F.copy()
    ksp.solve(F, X) # Inversion of the matrix

    Psol1.x.array[:offset] = X.array_r[:offset]
    Qsol1.x.array[:(len(X.array_r) - offset)] = X.array_r[offset:]

    Pav1[ii] = assemble_scalar(form(Psol1*ds(1)))
    ksp.destroy()
    X.destroy()
    Z.destroy()



surfarea = assemble_scalar(form(1*ds(1)))
k_output = 2*np.pi*freqvec/c0
Z_center = 1j*k_output* Pav1 / surfarea
#save_results(file_str, Z_center)


ax.plot(freqvec, Z_center.real, label = "alpha = 1 dim P = 4")
ax.grid(True)
ax.legend(loc='upper left')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel(r'$\sigma$')
plt.show()