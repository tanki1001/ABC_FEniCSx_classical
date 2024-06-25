# Modules importations
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
from operators_POO import (Mesh,
                        B1p, B2p, B3p, 
                        Loading, 
                        Simulation, 
                        import_frequency_sweep, import_COMSOL_result, store_results, plot_analytical_result_sigma,
                        least_square_err, compute_analytical_radiation_factor)
print("test2")
# Choice of the geometry among provided ones
geometry1 = 'cubic'
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

# To compare to COMSOL
comsol_data = True

if comsol_data:
    s = geometry
    frequency, results = import_COMSOL_result(s)

# Choice between using saved results or doing a new frequency sweep
from_data_b1p = True
from_data_b2p = True

#  Creation of a simulation with B1p
## Choice of a mesh, a loading, an operator -> Simulation
mesh_   = Mesh(1, side_box, radius, lc, geo_fct)
loading = Loading(mesh_)
ope1    = B1p(mesh_)
simu1   = Simulation(mesh_, ope1, loading)

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

# Creation a simulation with new operator B2p
mesh_   = Mesh(2, side_box, radius, lc, geo_fct)
loading = Loading(mesh_)
ope2    = B2p(mesh_)
simu2   = Simulation(mesh_, ope2, loading)

if from_data_b2p:
    s1 = 'FOM_b2p'
    s  = s1 + '_' + geometry
    freqvec2, PavFOM2 = import_frequency_sweep(s)
else :
    freqvec2 = freqvec
    PavFOM2 = simu2.FOM(freqvec2)
    s1 = 'FOM_b2p'
    s  = s1 + '_' + geometry
    store_results(s, freqvec2, PavFOM2)

# Creation a simulation with new operator B2p

mesh_.set_deg(3)
ope3    = B3p(mesh_)
loading = Loading(mesh_)

simu3   = Simulation(mesh_, ope3, loading)
freqvec3 = np.arange(80, 2001, 20)
PavFOM3 = simu3.FOM(freqvec3)

# Plot of the results with matplotlib - so far impossible except with jupyterlab
fig, ax = plt.subplots(figsize=(16,9))
simu1.plot_radiation_factor(ax, freqvec1, PavFOM1, s = 'FOM_b1p')
simu2.plot_radiation_factor(ax, freqvec2, PavFOM2,  s = 'FOM_b2p')
#print(f'PavFOM2 : {PavFOM2}')
simu3.plot_radiation_factor(ax, freqvec3, PavFOM3,  s = 'FOM_b3p')
if comsol_data:
    ax.plot(frequency, results, c = 'black', label=r'$\sigma_{COMSOL}$')
    ax.legend()

plot_analytical_result = True
if plot_analytical_result:
    plot_analytical_result_sigma(ax, freqvec, radius)

Z_ana = compute_analytical_radiation_factor(freqvec, radius)
err_B1p = least_square_err(freqvec, Z_ana.real, freqvec1, simu1.compute_radiation_factor(freqvec1, PavFOM1).real)
print(f'For lc = {lc} - L2_err(B1p) = {err_B1p}')

err_B2p = least_square_err(freqvec, Z_ana.real, freqvec2, simu2.compute_radiation_factor(freqvec2, PavFOM2).real)
print(f'For lc = {lc} - L2_err(B2p) = {err_B2p}')

err_B3p = least_square_err(freqvec, Z_ana.real, freqvec3, simu3.compute_radiation_factor(freqvec3, PavFOM3).real)
print(f'For lc = {lc} - L2_err(B3p) = {err_B3p}')