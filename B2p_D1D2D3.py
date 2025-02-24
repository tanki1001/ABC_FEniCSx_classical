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

dimP = 1
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

running_test = False
if not(running_test):
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
if not(running_test):
    #plot_analytical_result_sigma(ax, freqvec, radius)

    ax.plot(freqvec, Z_center.real)
    ax.grid(True)
    ax.legend(loc='upper left')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(r'$\sigma$')


#############################################
print("------------------------------------------")
print("WCAWE")
print("------------------------------------------")
# Functions
def sub_matrix(Q, alpha):
    '''
    This function is to obtain the sub matrix need for the correction term (P_q_w)
    intput :
        Q     = PETScMatType : the matrix where the norms and the scalar products are stored
        start = int : start index, reality index
        end   = int : end index, reality index

    output : 
        submatrix = np.array() : sub matrix, as a numpy matrix, because the size will remain low
    '''
     
    row_is    = PETSc.IS().createStride(alpha - 1, first = 1, step=1)
    col_is    = PETSc.IS().createStride(alpha - 1, first = 1, step=1)
    submatrix = Q.createSubMatrix(row_is, col_is)

    row_is.destroy()
    col_is.destroy()

    submatrix = submatrix.getValues([i for i in range(alpha - 2)], [i for i in range(alpha - 2)])
    return submatrix

def PQ2(alpha, Q):
    '''
    Correction term function.
    input :
        Q     = PETScMatType : the matrix where the norms and the scalar products are stored
        alpha = int : reality value
        beta  = int : reality value
        omega = int : starting point of the product

    output :
        P_q_w = PETScMatType : correction term
    '''

    P_q   = np.identity(alpha - 2) #create the identity matrix M*M with M = alpha - beta
    sub_Q = sub_matrix(Q, alpha)
    sub_Q = np.linalg.inv(sub_Q)
    P_q   = np.dot(P_q, sub_Q)
    # The following lignes convert the result to a PETSc type
    P_q_2 = PETSc.Mat().create()
    P_q_2.setSizes(P_q.shape, P_q.shape)
    P_q_2.setType("seqdense")  
    P_q_2.setFromOptions()
    P_q_2.setUp()

    for i in range(P_q.shape[0]):
        P_q_2.setValues(i, [j for j in range(P_q.shape[1])], P_q[i], PETSc.InsertMode.INSERT_VALUES)   
    P_q_2.assemble()
    return P_q_2

# Choice of the initial frequency
f0 = 1000
omega0 = 2*np.pi*f0
k0 = omega0/c0

# Compute of the derivatives of Z
Z0 = D1 + 1j*k0*D2 - k0**2*D3
Z1 = 2j*np.pi/c0*D2 - 4*np.pi/c0*k0*D3
Z2 = -8*np.pi**2/c0**2*D3

# Choice of the number of vector in the basis
N = 50

## Computation of the basis V
# Create the solver
ksp = PETSc.KSP().create()
ksp.setOperators(Z0)
ksp.setType("gmres")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")

# Initilization of the Q matrix
Q = PETSc.Mat().create()
Q.setSizes((N, N))  
Q.setType("seqdense")  
Q.setFromOptions()
Q.setUp()

# Compute of the first vector
v1 = F.copy()
ksp.solve(F, v1)

# Fill the Q matrix 
norm_v1 = v1.norm()
print(f' norm 1st vector : {norm_v1}')
v1.normalize()
Q.setValue(0, 0, norm_v1)
Q.assemble()

# Initilization of the basis V
size_v1 = v1.getSize()
Vn = PETSc.Mat().create()
Vn.setSizes((size_v1, N))  
Vn.setType("seqdense")  
Vn.setFromOptions()
Vn.setUp()    
Vn.setValues([i for i in range(size_v1)], 0, v1, PETSc.InsertMode.INSERT_VALUES) #Vn[0] = v1
Vn.assemble()

if N>=2:
    # Compute of the second vector
    rhsv2 = Z0.createVecLeft()
    Z1.mult(v1, rhsv2)
    rhsv2 = -rhsv2
    v2 = rhsv2.copy()
    ksp.solve(rhsv2, v2)
    rhsv2.destroy()
    norm_v2 = v2.norm()
    print(f' norm 2nd vector : {norm_v2}')
    Q.setValue(1, 1, norm_v2)
    Q.setValue(0, 1, v2.dot(v1))
    Q.assemble()
    v2  = v2 - v2.dot(v1) * v1
    v2.normalize()

    Vn.setValues([i for i in range(size_v1)], 1, v2, PETSc.InsertMode.INSERT_VALUES) #Vn[1] = v2
    Vn.assemble()
print(f'n = 2 | Q : {Q.view()}')
# Compute the rest of the basis
for n in range(3, N+1):
    print(f'Compute v{n}')

    # n is a reality index, to compute the third vector n = 3

    rhsvn = Z0.createVecLeft()
    vn_1 = Vn.getColumnVector(n-1   -1) # at the first interation n = 3, we need v2, and we get it with Vn.getColumnVector(n-1   -1) = Vn.getColumnVector(1)
    Z1.mult(vn_1, rhsvn) # rhsvn = Z1*vn_1
    if n == 3 :
        print("Test on vn_1 = v2")
        print(f'norm of vn_1 : {vn_1.norm()}')
        print(vn_1.getArray()[10 : 20]) 
        print(f'norm of v2 : {v2.norm()}')
        print(v2.getArray()[10 : 20]) 

    PQ2Value = PQ2(n, Q).getColumnVector(n-2   - 1)

    row_is = PETSc.IS().createStride(Vn.getSize()[0], first=0, step=1)
    col_is = PETSc.IS().createStride(n-2, first=0, step=1)
    
    Vn_2       = Vn.createSubMatrix(row_is, col_is)
    
    Vn_2       = Z2.matMult(Vn_2) # Vn_2 = Z2 * Vn_2
    Vn_2_P_q_2 = Vn_2.createVecLeft()
    Vn_2.mult(PQ2Value, Vn_2_P_q_2)
    rhsvn = - rhsvn - Vn_2_P_q_2

    vn = Vn.createVecLeft()
    ksp.solve(rhsvn, vn)

    rhsvn.destroy()
    row_is.destroy()
    col_is.destroy()
    Vn_2.destroy()
    Vn_2_P_q_2.destroy()
    PQ2Value.destroy()
    vn_1.destroy()

    norm_vn = vn.norm()
    print(f' norm {n}th vector : {norm_vn}')

    for i in range(n):
        # i is a programming index
        if i == n-1:
            Q.setValue(i, i, norm_vn) # Carefull it will be the place i+1. Q.setValue(2,3,7) will put 7 at the place (3,4)
        else:
            v_i = Vn.getColumnVector(i)     # Careful, asking for the vector i will give the (i+1)th reality vector
            Q.setValue(i, n-1, vn.dot(v_i)) # Carefull the function vn.dot(v_i) does the scalar product between vn and the conjugate of v_i
            v_i.destroy()
    Q.assemble()

    ## Gram-schmidt
    for i in range(n):
        # i is a programming index
        v_i = Vn.getColumnVector(i)
        vn  = vn - vn.dot(v_i) * v_i
        v_i.destroy()
    vn.normalize()
    Vn.setValues([i for i in range(size_v1)], n-1, vn, PETSc.InsertMode.INSERT_VALUES) # Careful, setValues(ni, nj, nk) considers indices as indexed from 0. Vn.setValues([2,4,9], [4,5], [[10, 11],[20, 21], [31,30]]) will change values at (3,5) = 10, (3, 6) = 11, (5, 5) = 20 ... #vn has been computed, to append it at the nth place in the base, we set up the (n-1)th column
    Vn.assemble()

    vn.destroy()
Q.destroy()
print("The basis has been computed")

# Compute the reduced matrices
Vn_T = Vn.duplicate()
Vn.copy(Vn_T)
Vn_T.hermitianTranspose()
Vn_T.assemble()
print(f'Vn size : {Vn.getSize()}')
print(f'Vn_T size : {Vn_T.getSize()}')

D1r = Vn_T.matMult(D1) 
D1r = D1r.matMult(Vn)
D1r.assemble()
print(f'D1r size : {D1r.getSize()}')

D2r = Vn_T.matMult(D2) 
D2r = D2r.matMult(Vn)
D2r.assemble()
print(f'D2r size : {D2r.getSize()}')

D3r = Vn_T.matMult(D3)
D3r = D3r.matMult(Vn)
D3r.assemble()
print(f'D3r size : {D3r.getSize()}')

Fr = D1r.createVecLeft()
Vn_T.mult(F, Fr) # Fn = Vn_T * F
print(f'Fr size : {Fr.getSize()}')

P, Q = mesh_.fonction_spaces()
Pav1         = np.zeros(freqvec.size, dtype=np.complex128)
Psol1, Qsol1 = Function(P), Function(Q)
offset       = P.dofmap.index_map.size_local * P.dofmap.index_map_bs # This is a bit obscur so far

for ii in tqdm(range(freqvec.size)):
    freq = freqvec[ii]
    omega0 = 2*np.pi*freq
    k0 = omega0/c0

    Zr = D1r + 1j*k0*D2r - k0**2*D3r
    #print(f'Type Zr : {(Zr.getType())}')
    Zr.convert("seqaij")
    #Z.assemble() # <- I'm not sure that's necessary | To test : try to do the simulation with and without to see if it leads to the same results
    if ii == 0:
        print(f"Size of the global reduced matrix: {Zr.getSize()}")

    # Solve
    ksp = PETSc.KSP().create()
    ksp.setOperators(Zr)
    ksp.setType("gmres") # Solver type 
    ksp.getPC().setType("lu") # Preconditionner type
    ksp.getPC().setFactorSolverType("mumps") # Various type of previous objects are available, and different tests have to be performed to find the best. Normaly this configuration provides best results

    alpha = Fr.copy()
    ksp.solve(Fr, alpha) # Inversion of the matrix

    X = F.copy()
    Vn.mult(alpha, X) # Projection back to the global solution

    Psol1.x.array[:offset] = X.array_r[:offset]
    Qsol1.x.array[:(len(X.array_r) - offset)] = X.array_r[offset:]

    Pav1[ii] = assemble_scalar(form(Psol1*ds(1)))
    ksp.destroy()
    X.destroy()
    Zr.destroy()

surfarea = assemble_scalar(form(1*ds(1)))
k_output = 2*np.pi*freqvec/c0
Z_center = 1j*k_output* Pav1 / surfarea
#save_results(file_str, Z_center)

#fig, ax = plt.subplots(figsize=(16,9))

ax.plot(freqvec, Z_center.real)
ax.grid(True)
ax.legend(loc='upper left')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel(r'$\sigma$')
ax.set_ylim(0,1.75)
plt.show()


