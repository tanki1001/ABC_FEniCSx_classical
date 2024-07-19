import numpy as np
from scipy import special
from geometries import cubic_domain, spherical_domain, half_cubic_domain, broken_cubic_domain
from postprocess import relative_errZ,import_FOM_result
from dolfinx.fem import (form, Function, FunctionSpace, petsc)
import petsc4py
from petsc4py import PETSc
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm

from operators_POO import Mesh, B1p, Loading, Simulation

geometry1 = 'spherical'
geometry2 = 'small'
geometry  = geometry1 + '_'+ geometry2

if   geometry2 == 'small':
    side_box = 0.11
    lc       = 6e-3
elif geometry2 == 'large':
    side_box = 0.40
    lc       = 2e-2
else :
    print("Enter your own side_box and mesh size in the code")
    side_box = 0.11
    lc       = 8e-3 #Typical mesh size : Small case : 8e-3 Large case : 2e-3

radius   = 0.1

rho0 = 1.21
c0   = 343.8

freqvec = np.arange(80, 2001, 20)

mesh_    = Mesh(1, side_box, radius, lc, cubic_domain)
_, ds, _ = mesh_.integral_mesure()


loading        = Loading(mesh_)
list_coeff_F_j = loading.deriv_coeff_F(0)

from operators_POO import B1p
ope1  = B1p(mesh_)
simu1 = Simulation(mesh_, ope1, loading)

from operators_POO import store_results, import_frequency_sweep
from_data_b1p = True
if from_data_b1p:
    s1 = 'FOM_b1p'
    s = s1 + '_' + geometry
    freqvec1, PavFOM1 = import_frequency_sweep(s)
else :
    freqvec1 = freqvec
    PavFOM1 = simu1.FOM(freqvec1)
    s1 = 'FOM_b1p'
    s = s1 + '_' + geometry
    store_results(s, freqvec, PavFOM1)


from operators_POO import SVD_ortho

list_N = [10]
list_freq = [1000]
t1   = time()
Vn   = simu1.merged_WCAWE(list_N, list_freq)
t2   = time()
print(f'WCAWE CPU time  : {t2 -t1}')
Vn = SVD_ortho(Vn)
t3 = time()
print(f'SVD CPU time  : {t3 -t2}')
PavWCAWE1 = simu1.moment_matching_MOR(Vn, freqvec)
t4       = time()
print(f'Whole CPU time  : {t4 -t1}')

from operators_POO import least_square_err, compute_analytical_radiation_factor

err_B1p_wcawe = least_square_err(freqvec, simu1.compute_radiation_factor(freqvec1, PavFOM1).real, freqvec1, simu1.compute_radiation_factor(freqvec1, PavWCAWE1).real)
print(f'For list_N = {list_N} - L2_err(B1p_wcawe) = {err_B1p_wcawe}')

if False:
    fig, ax = plt.subplots()
    simu1.plot_radiation_factor(ax, freqvec, PavFOM1, s = 'FOM_b1p')
    simu1.plot_radiation_factor(ax, freqvec, PavWCAWE, s = 'WCAWE')
    ax.set_ylim(0, 2)
    plt.savefig("WCAWE_b1p.png")

from operators_POO import B2p
mesh_.set_deg(2)
ope2       = B2p(mesh_)
loading        = Loading(mesh_)

simu2 = Simulation(mesh_, ope2, loading)

from_data_b2p = True

if from_data_b2p:
    s1 = 'FOM_b2p'
    s = s1 + '_' + geometry
    freqvec2, PavFOM2 = import_frequency_sweep(s)
else :
    freqvec2 = freqvec
    PavFOM2 = simu2.FOM(freqvec2)
    s1 = 'FOM_b2p'
    s = s1 + '_' + geometry
    store_results(s, freqvec, PavFOM2)

list_N = [10]
list_freq = [1750]

t1   = time()
Vn   = simu2.merged_WCAWE(list_N, list_freq)
t2   = time()
print(f'WCAWE CPU time  : {t2 -t1}')

Vn = SVD_ortho(Vn)
t3 = time()
print(f'SVD CPU time  : {t3 -t2}')
PavWCAWE2 = simu2.moment_matching_MOR(Vn, freqvec2)
t4 = time()
print(f'Whole CPU time  : {t4 -t1}')

err_B2p_wcawe = least_square_err(freqvec, simu2.compute_radiation_factor(freqvec2, PavFOM2).real, freqvec2, simu2.compute_radiation_factor(freqvec1, PavWCAWE2).real)
print(f'For list_N = {list_N} - L2_err(B2p_wcawe) = {err_B2p_wcawe}')

if True:
    fig, ax = plt.subplots()
    simu2.plot_radiation_factor(ax, freqvec2, PavFOM2, s = 'FOM_b2p')
    simu2.plot_radiation_factor(ax, freqvec2, PavWCAWE2, s = 'WCAWE')
    ax.set_ylim(0, 1.5)
    plt.title(f'list_N = {list_N}')
    plt.savefig("WCAWE_b2p.png")

from operators_POO import B3p
mesh_.set_deg(3)
ope3       = B3p(mesh_)
loading    = Loading(mesh_)

simu3 = Simulation(mesh_, ope3, loading)

from_data_b3p = True

if from_data_b3p:
    s1 = 'FOM_b3p'
    s = s1 + '_' + geometry
    freqvec3, PavFOM3 = import_frequency_sweep(s)
else :
    freqvec3 = freqvec
    PavFOM3 = simu3.FOM(freqvec3)
    s1 = 'FOM_b3p'
    s = s1 + '_' + geometry
    store_results(s, freqvec, PavFOM3)

list_N = [10]
list_freq = [1750]

t1   = time()
Vn   = simu3.merged_WCAWE(list_N, list_freq)
t2   = time()
print(f'WCAWE CPU time  : {t2 -t1}')

Vn = SVD_ortho(Vn)
t3 = time()
print(f'SVD CPU time  : {t3 -t2}')
PavWCAWE3 = simu3.moment_matching_MOR(Vn, freqvec3)
t4 = time()
print(f'Whole CPU time  : {t4 -t1}')

err_B3p_wcawe = least_square_err(freqvec, simu3.compute_radiation_factor(freqvec3, PavFOM3).real, freqvec3, simu3.compute_radiation_factor(freqvec3, PavWCAWE3).real)
print(f'For list_N = {list_N} - L2_err(B3p_wcawe) = {err_B3p_wcawe}')

if True:
    fig, ax = plt.subplots()
    simu3.plot_radiation_factor(ax, freqvec3, PavFOM3, s = 'FOM_b3p')
    simu3.plot_radiation_factor(ax, freqvec3, PavWCAWE3, s = 'WCAWE')
    ax.set_ylim(0, 1.5)
    plt.title(f'list_N = {list_N}')
    plt.savefig("WCAWE_b3p.png")