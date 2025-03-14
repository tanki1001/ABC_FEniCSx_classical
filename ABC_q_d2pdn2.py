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
import ufl
import petsc4py
from petsc4py import PETSc
import slepc4py.SLEPc as SLEPc

from geometries import cubic_domain, spherical_domain, half_cubic_domain, broken_cubic_domain
from operators_POO import (Mesh, Loading, plot_analytical_result_sigma)

def tangential_proj(u, n):
    return (ufl.Identity(n.ufl_shape[0]) - ufl.outer(n, n)) * u


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

# Simulation parameters
radius  = 0.1                               # Radius of the baffle
rho0    = 1.21                              # Density of the air
c0      = 343.8                             # Speed of sound in air
freqvec = np.arange(80, 2001, 20)           # List of the frequencies

dimP = 2
dimQ = 2

mesh_            = Mesh(dimP, dimQ, side_box, radius, lc, geo_fct)
submesh          = mesh_.submesh
entity_maps_mesh = mesh_.entity_maps_mesh

dx, ds, dx1 = mesh_.integral_mesure()

P, Q = mesh_.fonction_spaces()
xref = mesh_.xref
mesh = mesh_.mesh

p, v = TrialFunction(P), TestFunction(P)
q, u = TrialFunction(Q), TestFunction(Q)

f    = inner(1, v) * ds(1)
zero = inner(Constant(submesh, PETSc.ScalarType(0)), u) * dx1
f = [form(f), form(zero)]
F = petsc.assemble_vector_nest(f)

k = inner(grad(p), grad(v))
m = inner(p, v)

fx1 = Function(P)
fx1.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))

n   = FacetNormal(mesh)
dp  = inner(grad(p), n) # dp/dn = grad(p) * n

Pav1   = np.zeros(freqvec.size, dtype=np.complex128)
Psol1  = Function(P)
offset = P.dofmap.index_map.size_local * P.dofmap.index_map_bs # This is a bit obscur so far


for ii in tqdm(range(freqvec.size)):
    freq = freqvec[ii]
    omega0 = 2*np.pi*freq
    k0 = omega0/c0


    L1 = 2j*k0 + 4*fx1
    L2 = 2*fx1**2 - k0**2 + 4j*k0*fx1

    c = inner(L2/L1 * p, v)

    e = inner(1/L1 * q, v)
    
    d1 = inner(L1 * dp, u)
    d2 = inner(L2 * p, u)
    
    g = inner(q, u)


    z_00 = form(k * dx - k0**2 * m * dx + c * ds(3))
    z_01 = form(e*ds(3), entity_maps=entity_maps_mesh)
    z_10 = form(d1*ds(3) + d2*ds(3), entity_maps=entity_maps_mesh)
    z_11 = form(g*dx1)

    z = [[z_00, z_01],
        [z_10, z_11]]
    
    Z = petsc.assemble_matrix_block(z)
    Z.assemble()

    if ii == 0:
        print(f"Size of the global matrix: {Z.getSize()}")

    # Solve
    ksp = PETSc.KSP().create()
    ksp.setOperators(Z)
    ksp.setType("gmres") # Solver type 
    ksp.getPC().setType("lu") # Preconditionner type
    ksp.getPC().setFactorSolverType("mumps") # Various type of previous objects are available, and different tests have to be performed to find the best. Normaly this configuration provides best results

    X = F.copy()
    ksp.solve(F, X) # Inversion of the matrix

    Psol1.x.array[:offset] = X.array_r[:offset]

    Pav1[ii] = assemble_scalar(form(Psol1*ds(1)))
    ksp.destroy()
    X.destroy()
    Z.destroy()

surfarea = assemble_scalar(form(1*ds(1)))
k_output = 2*np.pi*freqvec/c0
Z_center = 1j*k_output* Pav1 / surfarea

fig, ax = plt.subplots(figsize=(16,9))

plot_analytical_result_sigma(ax, freqvec, radius)

ax.plot(freqvec, Z_center.real)
ax.grid(True)
ax.legend(loc='upper left')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel(r'$\sigma$')
plt.show()



