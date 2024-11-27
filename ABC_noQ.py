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
import petsc4py
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
dimQ = 20

file_str = 'b2p_' + geometry + "_" + str(dimP) + ".txt"

mesh_    = Mesh(dimP, dimQ, side_box, radius, lc, geo_fct)

dx, ds, _ = mesh_.integral_mesure()
P, _ = mesh_.fonction_spaces()
xref = mesh_.xref
mesh = mesh_.mesh

p, v = TrialFunction(P), TestFunction(P)

f = inner(1, v) * ds(1)

k = inner(grad(p), grad(v))
m = inner(p, v)

fx1 = Function(P)
fx1.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))

n   = FacetNormal(mesh)
dp  = inner(grad(p), n) # dp/dn = grad(p) * n
ddp = inner(grad(dp), n) # d^2p/dn^2 = grad(dp/dn) * n = grad(grad(p) * n) * n
c_1 = inner(ddp, v)
c_2 = inner(p, v)

Pav1    = np.zeros(freqvec.size, dtype=np.complex128)
Psol1 = Function(P)
for ii in tqdm(range(freqvec.size)):
    freq = freqvec[ii]
    omega0 = 2*np.pi*freq
    k0 = omega0/c0
    d_0 = 2j*k0 + 2*fx1
    d_1 = 2*fx1**2 - k0**2 + 4j*k0*fx1
    coeff_1 = 1/d_0
    coeff_2 = d_1/d_0
    c = coeff_1*c_1 + coeff_2*c_2

    a = k * dx - k0**2 * m * dx + c*ds(3)

    problem = LinearProblem(a, f, petsc_options={"ksp_type": "gmres",
                                            "pc_type": "lu",
                                            "pc_factor_mat_solver_type": "mumps"})

    p_solution = problem.solve()
    Pav1[ii] = assemble_scalar(form(p_solution*ds(1)))

surfarea = assemble_scalar(form(1*ds(1)))
k_output = 2*np.pi*freqvec/c0
Z_center = 1j*k_output* Pav1 / surfarea
#save_results(file_str, Z_center)

fig, ax = plt.subplots(figsize=(16,9))

plot_analytical_result_sigma(ax, freqvec, radius)

ax.plot(freqvec, Z_center.real)
ax.grid(True)
ax.legend(loc='upper left')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel(r'$\sigma$')
plt.show()



