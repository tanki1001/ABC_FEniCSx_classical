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
from operators_POO import (Mesh,
                        B1p, B2p, B3p, 
                        B2p_modified_r, B3p_modified_r,
                        B2p_modified_r6,
                        B2p_modified_dp_dq,
                        B2p_modified_mixB1p,
                        Loading, 
                        Simulation, 
                        import_frequency_sweep, import_COMSOL_result, store_results, store_resultsv2, store_resultsv3, plot_analytical_result_sigma,
                        least_square_err, compute_analytical_radiation_factor)

# Choice of the geometry among provided ones
geometry1 = 'cubic'
geometry2 = 'small'
geometry  = geometry1 + '_'+ geometry2

if   geometry2 == 'small':
    side_box = 0.11
    lc       = 8e-3
elif geometry2 == 'large':
    side_box = 0.40
    lc       = 1e-2
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
freqvec = np.linspace(80, 2000, 1)           # List of the frequencies



def main_cond_fct(  dimP,
                    dimQ,
                    str_ope, 
                    freqvec, 
                    geo_fct,
                    ax_sv):
    
    mesh_    = Mesh(dimP, dimQ, side_box, radius, lc, geo_fct)
    loading  = Loading(mesh_)

    if str_ope == "b1p":
        ope = B1p(mesh_)
    elif str_ope == "b2p":
        ope = B2p(mesh_)
    elif str_ope == "b2p_modified_r":
        ope = B2p_modified_r(mesh_)
    elif str_ope == "b2p_modified_r6":
        ope = B2p_modified_r6(mesh_)
    elif str_ope == "b2p_modified_dp_dq":
        ope = B2p_modified_dp_dq(mesh_)
    elif str_ope == "b2p_modified_mixb1p":
        ope = B2p_modified_mixB1p(mesh_)
    elif str_ope == "b3p":
        ope = B3p(mesh_)
    elif str_ope == "b3p_modified_r":
        ope = B3p_modified_r(mesh_)
    else:
        print("Operator doesn't exist")
        return

    simu    = Simulation(mesh_, ope, loading)

    simu.plot_condV3(freqvec, ax_sv, str_ope)


    


if True:
    fig, ax_sv = plt.subplots(nrows = 1, ncols = 2, figsize = (16, 9))

    dimP = 2
    dimQ = dimP
    str_ope = 'b2p'
    main_cond_fct(  
                dimP,
                dimQ,
                str_ope, 
                freqvec, 
                geo_fct, 
                ax_sv[0]

    )


    plt.show()