import numpy as np
from .geometries import cubic_domain, spherical_domain, half_cubic_domain, broken_cubic_domain
from time import time


from .operators_POO import (Mesh, Loading, Simulation,
                        B1p, B2p, B3p,
                        B2p_modified_r, B3p_modified_r,
                        B2p_modified_r6,
                        B2p_modified_mixB1p,
                        SVD_ortho,
                        store_results_wcawev2,
                        get_wcawe_param)

def main_wcawe(
        geo,
        case,
        str_ope,
        lc,
        dimP,
        dimQ,
        frequencies,
        n_values
):
    
    if   geo == 'cubic':
        geo_fct = cubic_domain
    elif geo == 'spherical':
        geo_fct = spherical_domain
    elif geo == 'half_cubic':
        geo_fct = half_cubic_domain
    elif geo == 'broken_cubic':
        geo_fct = broken_cubic_domain
    else :
        print("WARNING : May you choose an implemented geometry")

    if   case == 'small':
        side_box = 0.11
    elif case == 'large':
        side_box = 0.40
    else :
        print("Enter your own side_box and mesh size in the code")

    radius   = 0.1
    mesh_    = Mesh(dimP, dimQ, side_box, radius, lc, geo_fct)

    if str_ope == "b1p":
        ope = B1p(mesh_)
    elif str_ope == "b2p":
        ope = B2p(mesh_)
    elif str_ope == "b2p_modified_r":
        ope = B2p_modified_r(mesh_)
    elif str_ope == "b2p_modified_r6":
        ope = B2p_modified_r6(mesh_)
    elif str_ope == "b2p_modified_mixb1p":
        ope = B2p_modified_mixB1p(mesh_)
    elif str_ope == "b3p":
        ope = B3p(mesh_)
    elif str_ope == "b3p_modified_r":
        ope = B3p_modified_r(mesh_)
    else:
        print("Operator doesn't exist")
        return

    loading = Loading(mesh_)
    simu    = Simulation(mesh_, ope, loading)

    freqvec_fct = np.arange(80, 2001, 20) 

    t1   = time()
    Vn   = simu.merged_WCAWE(n_values, frequencies)
    t2   = time()
    print(f'WCAWE CPU time  : {t2 -t1}')

    #Vn = SVD_ortho(Vn)
    t3 = time()
    print(f'SVD CPU time  : {t3 -t2}')
    PavWCAWE_fct = simu.moment_matching_MOR(Vn, freqvec_fct)
    t4 = time()
    print(f'Whole CPU time  : {t4 -t1}')
    
    list_s = [geo, case, str_ope, str(lc), str(dimP), str(dimQ)]
    store_results_wcawev2(list_s, frequencies, n_values, freqvec_fct, PavWCAWE_fct, simu)


