# Modules importations
import numpy as np
from .geometries import cubic_domain, spherical_domain, half_cubic_domain, broken_cubic_domain, ellipsoidal_domain, curvedcubic_domain
from .operators_POO import (Mesh,
                        B1p, B2p, B3p, 
                        B2p_modified_r, B3p_modified_r,
                        B2p_modified_r6,
                        B2p_modified_dp_dq,
                        B2p_modified_mixB1p,
                        Loading, 
                        Simulation, 
                        store_resultsv4)




def main_ABC(geo,
             case,
             str_ope,
             lc,
             dimP,
             dimQ,
             freqvec = np.arange(80, 2001, 20)):
    
    # Choice of the geometry among provided ones
    geometry  = geo + '_'+ case

    if   case == 'small':
        side_box = 0.11
    elif case == 'large':
        side_box = 0.40
    else :
        print("Something wrong happened in the main_ABC fct (classical)")

    if   geo == 'cubic':
        geo_fct = cubic_domain
    elif geo == 'spherical':
        geo_fct = spherical_domain
    elif geo == 'half_cubic':
        geo_fct = half_cubic_domain
    elif geo == 'broken_cubic':
        geo_fct = broken_cubic_domain
    elif geo == 'ellipsoidal':
        geo_fct = ellipsoidal_domain
    elif geo == 'curvedcubic':
        geo_fct = curvedcubic_domain
    else :
        print("WARNING : May you choose an implemented geometry")

    radius  = 0.1                               # Radius of the baffle
    
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

    simu = Simulation(mesh_, ope, loading)

    if "modified" in str_ope:
        s1 = 'modified_ope/'
    else:
        s1 = 'classical/classical_'
    
    s2 = geometry + '_' + str_ope + '_' + str(lc) + '_' + str(dimP) + '_' + str(dimQ)
    s1 += s2
    
    PavFOM = simu.FOM(freqvec)
    store_resultsv4(s1, freqvec, PavFOM, simu)
    z_center = simu.compute_radiation_factor(freqvec, PavFOM)

    return freqvec, z_center


if __name__ == "__main__":
    print('Hello world')