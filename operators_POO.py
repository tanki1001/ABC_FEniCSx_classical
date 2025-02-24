import os
from pathlib import Path
import numpy as np
from math import factorial
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
from ufl import (TestFunction, TrialFunction, TrialFunctions,
                 dx, grad, inner, Measure, variable, FacetNormal, CellNormal)
import ufl
import petsc4py
from petsc4py import PETSc
import slepc4py.SLEPc as SLEPc

rho0   = 1.21
c0     = 343.8
source = 1

fr = symbols('fr')
k0 = 2*np.pi*fr/c0


class Mesh:
    

    def __init__(self, degP, degQ, side_box, radius, lc, geometry, model_name = "no_name"):
        '''
        Constructor of the class Mesh. A Mesh is created from a function implemented in the module geometries. This class is not perfect, it only implements geometries of this case.
        input : 
            side_box   = float : 
            radius     = float : 
            lc         = float :
            geometry   = fonction :
            model_name = str : 

        output : 
            Mesh

        '''
        self.degP = degP
        self.degQ = degQ
        
        mesh_info, submesh_info = geometry(side_box, radius, lc, model_name)

        self.mesh         = mesh_info[0]
        self.mesh_tags    = mesh_info[1]
        self.mesh_bc_tags = mesh_info[2]
        self.xref         = mesh_info[3]
    
        self.submesh          = submesh_info[0]
        self.entity_maps_mesh = submesh_info[1]

    
    def integral_mesure(self):
        '''
        This function gives access to the integral operator over the mesh and submesh in Mesh instance
        input :

        output :
            dx  = Measure : integral over the whole domain
            ds  = Measure : integral over the tagged surfaces
            dx1 = Measure : integral over the whole subdomain

        '''
        mesh         = self.mesh
        mesh_tags    = self.mesh_tags
        mesh_bc_tags = self.mesh_bc_tags
    
        submesh = self.submesh

        dx  = Measure("dx", domain=mesh, subdomain_data=mesh_tags)
        ds  = Measure("ds", domain=mesh, subdomain_data=mesh_bc_tags)
        dx1 = Measure("dx", domain=submesh)
        
        return dx, ds, dx1

    def fonction_spaces(self, family = "Lagrange"):
        '''
        This function provide fonction spaces needed in the FEM. They are spaces where the test and trial functions will be declared.
        input : 
            family = str : family of the element

        output : 
            P = FunctionSpace : fonction space where the fonctions living in the acoutic domain will be declared
            Q = FonctionSpace : fonction space where the fonctions living in the subdomain will be declared
        '''
        degP    = self.degP
        degQ    = self.degQ 
        mesh    = self.mesh
        submesh = self.submesh
    
        P1 = element(family, mesh.basix_cell(), degP)
        P = functionspace(mesh, P1)

        Q1 = element(family, submesh.basix_cell(), degQ)
        Q = functionspace(submesh, Q1)
        
        return P, Q

    

class Simulation:

    def __init__(self, mesh, operator, loading):
        '''
        Constructor of the class Simulation. 
        input : 
            mesh     = Mesh
            operator = Operator
            loading  = Loading
        output :
            Simulation
        '''
        self.mesh     = mesh
        self.operator = operator
        self.loading  = loading

    def set_mesh(self, mesh):
        '''
        Setter to change the geometry on the one the same simulation will be run
        input : 
            mesh = Mesh : new Mesh obtained from a new geometry
        '''
        self.mesh = mesh

    def set_operator(self, ope):
        '''
        Setter to change the operator applied on the simulation
        input : 
            ope = Operator 
        '''
        self.operator = ope

    def set_loading(self, loading):
        '''
        Setter to change the loading applied on the simulation
        input : 
            loading = Loading 
        '''
        self.loading = loading
    
    # To edit
    def FOM(self, freq, frequency_sweep = True):

        if frequency_sweep and not(isinstance(freq, int)):
            print('Frequency sweep')
            return self.freq_sweep_FOM_newVersion(freq)
        elif isinstance(freq, int) and not(frequency_sweep):
            print('Singular frequency')
            return self.singular_frequency_FOM(freq)
    

    # To edit
    def freq_sweep_FOM_newVersion(self, freqvec):
        '''
        This function runs a frequency sweep on the simulation to obtain the acoustic pressure along the vibrating surface.
        input : 
            freqvec = np.arange() : frequency interval

        output :
            Pav1 = np.array() : pressure field obtained along the vibrating plate
            
        '''
        ope     = self.operator
        list_D = ope.list_D
        loading = self.loading
        Pav1    = np.zeros(freqvec.size, dtype=np.complex128)
        mesh    = self.mesh
        submesh = mesh.submesh
        
        P, Q         = mesh.fonction_spaces()
        Psol1, Qsol1 = Function(P), Function(Q)
        offset       = P.dofmap.index_map.size_local * P.dofmap.index_map_bs # This is a bit obscur so far
        v, u = TestFunction(P), TestFunction(Q)
        _, ds, dx1 = mesh.integral_mesure()

        #F = loading.F
        f = form(inner(1, v) * ds(1))
        zero = form(inner(Constant(submesh, PETSc.ScalarType(0)), u) * dx1)

        F = [f, zero]
        F = petsc.assemble_vector_nest(F)
        
        
        ksp = PETSc.KSP().create()
        ksp.setType("gmres") # Solver type 
        ksp.getPC().setType("lu") # Preconditionner type
        ksp.getPC().setFactorSolverType("mumps") # Various type of previous objects are available, and different tests have to be performed to find the best. Normaly this configuration provides best results

        for ii in tqdm(range(freqvec.size)):
            freq = freqvec[ii]
            k0 = 2*np.pi*freq/c0
            Z = list_D[0]
            for i in range(1, len(list_D)):
                Z = Z + (1j*k0)**i*list_D[i]
            if ii == 0:
                print(f"Size of the global matrix: {Z.getSize()}")
    
                
            # Solve
            
            ksp.setOperators(Z)
            X = F.copy()
            ksp.solve(F, X) # Inversion of the matrix
        
            Psol1.x.array[:offset] = X.array_r[:offset]
            Qsol1.x.array[:(len(X.array_r) - offset)] = X.array_r[offset:]
        
            Pav1[ii] = assemble_scalar(form(Psol1*ds(1)))
            X.destroy()
            Z.destroy()
            
        

        return Pav1
    
    # To test
    def plot_row_columns_norm(self, freq, s = ''):
        ope     = self.operator

        #list_coeff_Z_j = ope.deriv_coeff_Z(0)

        #Z = ope.dZj(freq, list_coeff_Z_j[0])
        list_D = ope.list_D
        k0 = 2*np.pi*freq/c0
        Z = list_D[0]
        for i in range(1, len(list_D)):
            Z = Z + (1j*k0)**i*list_D[i]

        list_row_norms    = row_norms(Z)
        list_column_norms = column_norms(Z)

        fig, (ax_row, ax_col) = plt.subplots(nrows=2, ncols=1, figsize = (16, 9))
        ax_row.bar(range(len(list_row_norms)), list_row_norms)
        ax_row.set_ylim([min(list_row_norms), max(list_row_norms)])
        ax_col.bar(range(len(list_column_norms)), list_column_norms)
        ax_col.set_ylim([min(list_column_norms), max(list_column_norms)])

        ax_row.set_title('rows')
        ax_col.set_title('colums')
        
        plt.savefig('curves/ABC_curves/rows_col_norm/' +s + f'_row_columns_norm_{freq}.png')
        print(s + f'_row_columns_norm_{freq}.png has been downloaded')
  
    # To test
    def plot_matrix_heatmap(self, freq, s = ''):
        
        ope     = self.operator

        #list_coeff_Z_j = ope.deriv_coeff_Z(0)

        #Z = ope.dZj(freq, list_coeff_Z_j[0])
        list_D = ope.list_D
        k0 = 2*np.pi*freq/c0
        Z = list_D[0]
        for i in range(1, len(list_D)):
            Z = Z + (1j*k0)**i*list_D[i]

        # Obtain the dimensions of the matrix
        m, n = Z.getSize()

        # Initialise a numpy matrix to store the coefficients
        #matrix_values = np.zeros((m, n), dtype = 'complex')

        # Lists to store indices and non-zero values (real and imaginary parts)
        rows = []
        cols = []
        real_values = []
        imag_values = []

        # Browse each row of the matrix
        for i in range(m):
            
            row_cols, row_values = Z.getRow(i)
            rows.extend([i] * len(row_cols))  # Repeat the row index for each non-zero value
            cols.extend(row_cols)
            # Extract the real and imaginary parts of the coefficients
            real_values.extend([val.real for val in row_values])
            imag_values.extend([val.imag for val in row_values])

        # Create a figure with two sub-graphs
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Draw the mapping of the real part
        scatter1 = ax1.scatter(cols, rows, c=real_values, cmap='viridis', marker='s', s=10)
        ax1.set_title('Real part Mapping')
        ax1.set_xlabel('Column index')
        ax1.set_ylabel('Row index')
        ax1.invert_yaxis()
        plt.colorbar(scatter1, ax=ax1, label='Real Part')

        # Draw the mapping of the imaginary part
        scatter2 = ax2.scatter(cols, rows, c=imag_values, cmap='plasma', marker='s', s=10)
        ax2.set_title('Imaginary part Mapping')
        ax2.set_xlabel('Column index')
        ax2.set_ylabel('Row index')
        ax2.invert_yaxis()
        plt.colorbar(scatter2, ax=ax2, label='Imaginary Part')

        # Plot graphs side by side
        plt.tight_layout()
        plt.savefig('curves/ABC_curves/heatmap/' +s + f'_colored_matrix_{freq}.png')
        print(s + f'_colored_matrix_{freq}.png has been downloaded')

    # To test
    def plot_cond(self, freqvec, s ='',):
        ope = self.operator
        
        list_condition_number = []
        
        fig, (ax_cn, ax_sv) = plt.subplots(nrows = 1, ncols = 2, figsize = (16, 9))
        list_D = ope.list_D

        for freq in freqvec:
            list_coeff_Z_j = ope.deriv_coeff_Z(0)
            #Z = ope.dZj(freq, list_coeff_Z_j[0])
            k0 = 2*np.pi*freq/c0
            Z = list_D[0]
            for i in range(1, len(list_D)):
                Z = Z + (1j*k0)**i*list_D[i]


            condition_number, list_sigma = get_cond_nb(Z)
            print(f'list_sigma = {list_sigma}')

            list_condition_number.append(condition_number)
            ax_sv.scatter([freq for _ in range(len(list_sigma))], list_sigma)
        
        
        ax_cn.plot(freqvec, list_condition_number, label = 'conditioning number')
        ax_cn.set_xlabel('Frequency')
        ax_cn.set_ylabel('Conditionning number')

        ax_sv.set_xlabel('Frequency')
        ax_sv.set_ylabel('sigma')

        ax_cn.legend()


        # Plot graphs side by side
        plt.tight_layout()
        plt.savefig('curves/ABC_curves/cond_curves/' + s + f'_svd.png')
        print(s + f'_svd.png has been downloaded')
    

    # This method uses an old way to assemble matrices. This version can be found in the method ope.get_listZ()
    # To test
    def plot_sv_listZ(self, s =''):
        ope              = self.operator
        entity_maps_mesh = ope.mesh.entity_maps_mesh

        listZ = ope.get_listZ()
        
        fig, ax = plt.subplots(layout='constrained',figsize = (16, 9))

        width = 0.05

        index_mat = 1

        for z in listZ:
            z_form = form(z, entity_maps=entity_maps_mesh)

            Z = petsc.assemble_matrix(z_form)
            Z.assemble()

            cond_nb, cond_nb_list = get_cond_nb(Z)

            print(f'Conditioning number of the {index_mat}th matrix: {cond_nb}')
                        
            for i in range(len(cond_nb_list)):
                if i == 0:
                    rects = ax.bar(index_mat - width/2*(len(cond_nb_list) -1 - i*2), cond_nb_list[i], width, label = str(cond_nb))
                else:
                    rects = ax.bar(index_mat - width/2*(len(cond_nb_list) -1 - i*2), cond_nb_list[i], width)
                ax.bar_label(rects, padding=3)
                

            index_mat += 1
        ax.set_xticks([i+1 for i in range(len(listZ))])  
        ax.legend() 
        plt.savefig('curves/ABC_curves/sv_listZ/'+s+'_plot_sv_listZ.png')
        print(s+'_plot_sv_listZ.png has been downloaded')

    # Already clean
    def singular_frequency_FOM(self, freq):
        '''
        This function runs a frequency sweep on the simulation to obtain the acoustic pressure along the vibrating surface.
        input : 
            freq = int : frequency 

        output :
            Psol1 = Function : pressure field in the whole acoustic domain
            Qsol1 = Function : derivate of pressure field along the boundaries
            
        '''
        ope     = self.operator
        list_D  = ope.list_D
        mesh    = self.mesh
        submesh = mesh.submesh
        
        P, Q         = mesh.fonction_spaces()
        Psol1, Qsol1 = Function(P), Function(Q)
        offset       = P.dofmap.index_map.size_local * P.dofmap.index_map_bs # This is a bit obscur so far
        v, u         = TestFunction(P), TestFunction(Q)
        
        _, ds, dx1 = mesh.integral_mesure()
        ksp = PETSc.KSP().create()
        ksp.setType("gmres") # Solver type 
        ksp.getPC().setType("lu") # Preconditionner type
        ksp.getPC().setFactorSolverType("mumps") # Various type of previous objects are available, and different tests have to be performed to find the best. Normaly this configuration provides best results


        f = form(inner(1, v) * ds(1))
        zero = form(inner(Constant(submesh, PETSc.ScalarType(0)), u) * dx1)

        F = [f, zero]
        F = petsc.assemble_vector_nest(F)
        F.assemble()
        
        k0 = 2*np.pi*freq/c0
        Z = list_D[0]
        for i in range(1, len(list_D)):
            Z = Z + (1j*k0)**i*list_D[i]
            print(f"Size of the global matrix: {Z.getSize()}")

            
        # Solve
        
        ksp.setOperators(Z)
        X = F.copy()
        ksp.solve(F, X) # Inversion of the matrix
    
        Psol1.x.array[:offset] = X.array_r[:offset]
        Qsol1.x.array[:(len(X.array_r) - offset)] = X.array_r[offset:]
    
        ksp.destroy()
        X.destroy()
        Z.destroy()
        F.destroy()

        #harry_plotter(P, Psol1, 'p', show_edges = True)  
        #harry_plotter(Q, Qsol1, 'q', show_edges = True)
        harry_plotterv2([P, Q], [Psol1, Qsol1], ['p', 'q'], show_edges = True)
            
        return Psol1, Qsol1
       
    # To edit
    def wcawe_newVersion(self, N, freq):
        '''
        One of the most complex function. This function implements the WCAWE model order reduction method.
        input :
            N    = int : nb of vector in the projection basis
            freq = int : interpolation point

        output : 
            Vn = PETScMat : projection basis
        '''
        submesh          = self.mesh.submesh
        ope              = self.operator
        list_D           = ope.list_D
        degZ             = len(list_D)
        loading          = self.loading

        P, Q        = self.mesh.fonction_spaces()
        _, ds, dx1 = self.mesh.integral_mesure() 
    
        v, u = TestFunction(P), TestFunction(Q)
        
        f = form(inner(1, v) * ds(1))
        zero = form(inner(Constant(submesh, PETSc.ScalarType(0)), u) * dx1)

        F = [f, zero]
        F = petsc.assemble_vector_nest(F)
        
        
        # The global matrix and its first derivative are needed to construct each vector, that's why they are computed
        # outside of the loop

        ## Compute the jth derivative of Z
        list_Zj = []
        for j in range(degZ):
            Zj = 0 # Weird way to initialize
            for i in range(j, degZ):
                coeff = (2j*np.pi/c0)**i*factorial(i)/factorial(i-j)*freq**(i-j)
                Zj = Zj + coeff*list_D[i]
            list_Zj.append(Zj)
        ### Create the solver
        ksp = PETSc.KSP().create()
        ksp.setOperators(list_Zj[0])
        ksp.setType("gmres")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")
        
        ### Create Q matrix
        Q = PETSc.Mat().create()
        Q.setSizes((N, N))  
        Q.setType("seqdense")  
        Q.setFromOptions()
        Q.setUp()       
        
        ### Obtain the first vector, its size will be needed to create the basis
        #F_0 = loading.F
        v1 = F.copy()
        ksp.solve(F, v1)
        
        norm_v1 = v1.norm()
        print(f'norm 1st vector : {norm_v1}')
        v1.normalize()
        Q.setValue(0, 0, norm_v1)
        size_v1 = v1.getSize()
    
        ### Create the empty basis
        Vn = PETSc.Mat().create()
        Vn.setSizes((size_v1, N))  
        Vn.setType("seqdense")  
        Vn.setFromOptions()
        Vn.setUp()    
        Vn.setValues([i for i in range(size_v1)], 0, v1, PETSc.InsertMode.INSERT_VALUES) #Vn[0] = v1
        
        
        
        for n in range(2, N+1):
            # BIG WARNING in this new version (D1D2D3 version from 21st of January), we consider the first sum equals to 0 because the loading
            # is frequency independent
            rhs3 = list_Zj[0].createVecLeft()
    
            for j in range(2, min(degZ, n)):
                # j is a reality index and represents the index in the sum
  
                # The second sum starts only at j = 2
                P_q_2        = P_Q_w(Q, n, j, 2)
                P_q_2_values = P_q_2.getColumnVector(n-j-1)
                
                row_is = PETSc.IS().createStride(Vn.getSize()[0], first=0, step=1)
                col_is = PETSc.IS().createStride(n-j, first=0, step=1)
                
                Vn_i       = Vn.createSubMatrix(row_is, col_is)
                Vn_i       = list_Zj[j].matMult(Vn_i) # Vn_i = Z_i * Vn_i
                Vn_i_P_q_2 = Vn_i.createVecLeft()
                Vn_i.mult(P_q_2_values, Vn_i_P_q_2)
                

                # Second sum
                rhs3 = rhs3 + Vn_i_P_q_2

                row_is.destroy()
                col_is.destroy()
                P_q_2.destroy()
                P_q_2_values.destroy()
                Vn_i.destroy()
                Vn_i_P_q_2.destroy()

            rhs2 = list_Zj[0].createVecLeft()
    
            vn_1 = Vn.getColumnVector(n-2)
            list_Zj[1].mult(vn_1, rhs2) # rhs2 = Z_1 * vn_1 
             
            rhs = - rhs2 - rhs3
            vn = Vn.createVecLeft()
            ksp.solve(rhs, vn)
            rhs.destroy()
            rhs2.destroy()
            rhs3.destroy()
            
            norm_vn = vn.norm()
            print(f' norm {n}th vector : {norm_vn}')
            
            for i in range(n):
                if i == n-1:
                    Q.setValue(i, i, norm_vn) # Carefull it will be the place i+1. Q.setValue(2,3,7) will put 7 at the place (3,4)
                else:
                    v_i = Vn.getColumnVector(i)     # Careful, asking for the vector i will give the (i+1)th reality vector
                    Q.setValue(i, n-1, vn.dot(v_i)) # Carefull the function vn.dot(v_i) does the scalar product between vn and the conjugate of v_i
                    v_i.destroy()
            Q.assemble()
            #print(Q.view())
            ## Gram-schmidt
            for i in range(n):
                v_i = Vn.getColumnVector(i)
                vn  = vn - vn.dot(v_i) * v_i
                v_i.destroy()
            vn.normalize()
            Vn.setValues([i for i in range(size_v1)], n-1, vn, PETSc.InsertMode.INSERT_VALUES) # Careful, setValues(ni, nj, nk) considers indices as indexed from 0. Vn.setValues([2,4,9], [4,5], [[10, 11],[20, 21], [31,30]]) will change values at (3,5) = 10, (3, 6) = 11, (5, 5) = 20 ... #vn has been computed, to append it at the nth place in the base, we set up the (n-1)th column
        ksp.destroy()
        
        Vn.assemble()
        return Vn

    def merged_WCAWE(self, list_N, list_freq):
        if len(list_N) != len(list_freq):
            print(f"WARNING : The list of nb vector values and the list of interpolated frequencies does not match: {len(list_N)} - {len(list_freq)}")

        size_Vn = sum(list_N)
        V1 = self.wcawe_newVersion(list_N[0], list_freq[0])
        size_V1 = V1.getSize()[0]
        Vn = PETSc.Mat().create()
        Vn.setSizes((size_V1, size_Vn))
        Vn.setType("seqdense")  
        Vn.setFromOptions()
        Vn.setUp()    
        for i in range(V1.getSize()[0]):
            for j in range(V1.getSize()[1]):
                Vn[i, j] = V1[i, j]
        count = list_N[0]
        for i in range (1,len(list_freq)):
            Vi = self.wcawe_newVersion(list_N[i], list_freq[i])
            for ii in range(Vi.getSize()[0]):
                for jj in range(Vi.getSize()[1]):
                    Vn[ii, count + jj] = Vi[ii, jj]
            count += list_N[i]
        Vn.assemble()
        return Vn

    def moment_matching_MOR(self, Vn, freqvec):
        '''
        This function does a frequency sweep of the reduced model where a moment matching method has been applied. It is basically the same function as FOM(), but two lines has been added to project the matrix and the vector force
        input :
            Vn      = PETScMatType : projection basis
            freqvec = np.arange() : frequency interval
        
        output :
            Pav1 = np.array() : pressure field obtained along the vibrating plate    
        '''
          
        ope     = self.operator
        list_D = ope.list_D
    
        
        
        mesh_   = self.mesh
        submesh = self.mesh.submesh

        Pav1         = np.zeros(freqvec.size, dtype=np.complex128)
        P, Q         = mesh_.fonction_spaces()
        Psol1, Qsol1 = Function(P), Function(Q)
        offset       = P.dofmap.index_map.size_local * P.dofmap.index_map_bs
         
        _, ds, dx1 = mesh_.integral_mesure()

        v, u = TestFunction(P), TestFunction(Q)
        

        f = form(inner(1, v) * ds(1))
        zero = form(inner(Constant(submesh, PETSc.ScalarType(0)), u) * dx1)

        F = [f, zero]
        F = petsc.assemble_vector_nest(F)

        list_Dr, Fr = listDr_matrices(list_D, F, Vn)

        for ii in tqdm(range(freqvec.size)):
            freq   = freqvec[ii]
            k0 = 2*np.pi*freq/c0
            Zr = list_Dr[0]
            for i in range(1, len(list_Dr)):
                Zr = Zr + (1j*k0)**i*list_Dr[i]
            if ii == 0:
                print(f"Size of the global reduced matrix: {Zr.getSize()}")

            #Zn, Fn = Zn_Fn_matrices(Z, F, Vn) # Reduction of Z and F
            Zr.convert("seqaij")              # Conversion of Zn to be solved with superlu and mumps
            
            # Solve
            ksp = PETSc.KSP().create()
            ksp.setOperators(Zr)
            ksp.setType("gmres")                       # Solver type 
            ksp.getPC().setType("lu")                  # Preconditionner type
            ksp.getPC().setFactorSolverType("mumps")   # Various type of previous objects are available, and different tests hae to be performed to find the best. Normaly this configuration provides best results
            
            alpha = Fr.copy()
            ksp.solve(Fr, alpha) # Inversion of the matrix
        
            X = F.copy()
            Vn.mult(alpha, X) # Projection back to the global solution
            
            
            Psol1.x.array[:offset]                    = X.array_r[:offset]
            Qsol1.x.array[:(len(X.array_r) - offset)] = X.array_r[offset:]
            # Qsol1 can provide information in the q unknown which is the normal derivative of the pressure field along the boundary. Qsol1 is not used in this contribution.
        
        
            Pav1[ii] = assemble_scalar(form(Psol1*ds(1)))
            ksp.destroy()
            X.destroy()
            Zr.destroy()

        return Pav1

    def compute_radiation_factor(self, freqvec, Pav):
        _, ds, _ = self.mesh.integral_mesure()
        surfarea = assemble_scalar(form(1*ds(1)))
        k_output = 2*np.pi*freqvec/c0
        Z_center = 1j*k_output* Pav / surfarea
        return Z_center

    def plot_radiation_factor(self, ax, freqvec, Pav, s = '', compute = True):
        
        if compute :
            Z_center = self.compute_radiation_factor(freqvec, Pav)
        else:
            Z_center = Pav
    
        ax.plot(freqvec, Z_center.real, label = s)
        ax.grid(True)
        ax.legend(loc='upper left')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel(r'$\sigma$')
    

       

class Operator(ABC):
    '''
    This method aims at being an abstract one. One will definied an implemented operator, and never use the followong constructor.
    '''
    def __init__(self, mesh):
        '''
        Constructor of the class Operator. The objective is to overwrite this constructor in order for the user to only use designed Operator.
        input : 
            mesh = Mesh : instance of the class Mesh, where the Operator is applied on

        output :
            Operator
        '''

        self.mesh        = mesh
        #self.list_Z      = None
        #self.list_coeffZ = None
        self.list_D      = None

    @abstractmethod
    def import_matrix(self, freq):
        pass

    # This method, using an old way to assemble matrices, serves the method Simulation.plot_sv_listZ
    @abstractmethod
    def get_listZ(self):
        pass

class B1p(Operator):

    def __init__(self, mesh):
        '''
        Constructor of the b1p operator.
        '''
        super().__init__(mesh)
        #self.list_Z, self.list_coeff_Z = self.b1p()
        self.list_D = self.b1p_newVersion()

  
    # To edit
    def b1p_newVersion(self):
        '''
        Create all the constant Form of the b1p operator on the given mesh/simulation and all the coeff of the global matrix, all the constant Form of the force block vector and the related coeff
    
        input :
            mesh_info    = List[]
            submesh_info = List[]
    
        output :
            list_Z       = np.array(Form) : List of the constant matrices of the b1p operator
            list_coeff_Z = np.array() : array of the coeff as sympy expression w.r.t 'fr' symbol
    
        '''
        mesh         = self.mesh.mesh
        submesh      = self.mesh.submesh
        xref         = self.mesh.xref

        entity_maps_mesh = self.mesh.entity_maps_mesh
    
        #deg         = self.deg
        P, Q        = self.mesh.fonction_spaces()
        dx, ds, dx1 = self.mesh.integral_mesure() 
    
        p, q = TrialFunction(P), TrialFunction(Q)
        v, u = TestFunction(P), TestFunction(Q)
        
        fx1 = Function(P)
        fx1.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))
        
        k = inner(grad(p), grad(v)) * dx
        m = inner(p, v) * dx
        c = inner(q, v)*ds(3)
        
        g1 = inner(fx1*p, u)*ds(3)
        g2 = inner(p, u)*ds(3)
        e  = inner(q, u)*dx1

        K = form(k)
        M = form(m)
        C = form(-c, entity_maps=entity_maps_mesh)
        
        G1 = form(g1, entity_maps=entity_maps_mesh)
        G2 = form(g2, entity_maps=entity_maps_mesh)
        
        E = form(e)

        D1 = [[K,  C],
              [G1, E]]
        D1 = petsc.assemble_matrix_block(D1)
        D1.assemble()
        

        D2_00 = form(inner(Constant(mesh, PETSc.ScalarType(0))*p, v) * dx)
        D2_01 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, v) * ds(3), entity_maps=entity_maps_mesh)
        D2_11 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, u) * dx1, entity_maps=entity_maps_mesh)
        D2 = [[D2_00, D2_01],
              [G2,    D2_11]]
        D2 = petsc.assemble_matrix_block(D2)
        D2.assemble()

        D3_01 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, v) * ds(3), entity_maps=entity_maps_mesh)
        D3_10 = form(inner(Constant(mesh, PETSc.ScalarType(0))*p, u) * ds(3), entity_maps=entity_maps_mesh)
        D3_11 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, u) * dx1)
        D3 = [[M,     D3_01],
              [D3_10, D3_11]]
        D3 = petsc.assemble_matrix_block(D3)
        D3.assemble()
    
        list_D       = np.array([D1, D2, D3])
    
        return list_D

    def get_listZ(self):
        mesh_tags    = self.mesh.mesh_tags
        mesh_bc_tags = self.mesh.mesh_bc_tags
        xref         = self.mesh.xref

        entity_maps_mesh = self.mesh.entity_maps_mesh
    
        #deg         = self.deg
        P, Q        = self.mesh.fonction_spaces()
        dx, ds, dx1 = self.mesh.integral_mesure() 
    
        p, q = TrialFunction(P), TrialFunction(Q)
        v, u = TestFunction(P), TestFunction(Q)
        
        fx1 = Function(Q)
        fx1.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))
        
        k = inner(grad(p), grad(v)) * dx
        m = inner(p, v) * dx
        c = inner(q, v)*ds(3)
        
        g1 = inner(p, u)*ds(3)
        g2 = inner(fx1*p, u)*ds(3)
        e  = inner(q, u)*dx1
    
        list_Z       = np.array([k, m, c, g1, g2, e])
        return list_Z


    def import_matrix(self, freq):

        list_D = self.b1p_newVersion()
        k0 = 2*np.pi*freq/c0
        Z = list_D[0]
        for i in range(1, len(list_D)):
            Z = Z + (1j*k0)**i*list_D[i]

        ope_str = 'b1p'
        ai, aj, av = Z.getValuesCSR()
        Zsp = csr_matrix((av, aj, ai))
        savemat('Z_'+ope_str+'_ref.mat', {'Z'+ope_str:Zsp})
    
        
class B2p(Operator):

    def __init__(self, mesh):
        '''

        Constructor of the b2p operator.
        '''
        super().__init__(mesh)
        #self.list_Z, self.list_coeff_Z = self.b2p()
        self.list_D = self.b2p_newVersion()
        

  
    # To edit
    def b2p_newVersion(self):
        '''
        Create all the constant Form of the b2p operator on the given mesh/simulation and all the coeff of the global matrix, all the constant Form of the force block vector and the related coeff
    
        input :
            mesh_info    = List[]
            submesh_info = List[]
    
        output :
            list_Z       = np.array(Form) : List of the constant matrices of the b1p operator
            list_coeff_Z = np.array() : array of the coeff as sympy expression w.r.t 'fr' symbol
    
        '''
        mesh    = self.mesh.mesh
        submesh = self.mesh.submesh
        xref    = self.mesh.xref

        entity_maps_mesh = self.mesh.entity_maps_mesh
        n                = FacetNormal(mesh) # Normal to the boundaries
        
        #deg        = self.deg
        P, Q        = self.mesh.fonction_spaces()
        dx, ds, dx1 = self.mesh.integral_mesure() 
    
        p, q = TrialFunction(P), TrialFunction(Q)
        v, u = TestFunction(P), TestFunction(Q)
        
        fx1_p = Function(P)
        fx1_p.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))

        fx1_q = Function(Q)
        fx1_q.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))
        
        k = inner(grad(p), grad(v)) * dx
        m = inner(p, v) * dx
        c = inner(q, v)*ds(3)
        
        dp  = inner(grad(p), n) # dp/dn = grad(p) * n
        ddp = inner(grad(dp), n) # d^2p/dn^2 = grad(dp/dn) * n = grad(grad(p) * n) * n
        g1  = inner(ddp, u)*ds(3)
        g2  = inner(p, u)*ds(3)
        g3  = inner(4*fx1_p*p, u)*ds(3)
        g4  = inner(2*fx1_p**2*p, u)*ds(3)

        e1  = inner(4*fx1_q*q, u)*dx1
        e2  = inner(2*q, u)*dx1

        K = form(k)
        M = form(m)
        C = form(-c, entity_maps=entity_maps_mesh)
        
        # The two following matrices are not created, their sum is
        #G1 = form(g1, entity_maps=entity_maps_mesh)
        #G4 = form(g4, entity_maps=entity_maps_mesh)
        
        G2 = form(g2, entity_maps=entity_maps_mesh)
        G3 = form(g3, entity_maps=entity_maps_mesh)
        
        G1_4 = form(g1 + g4, entity_maps=entity_maps_mesh)
        

        E1 = form(e1)
        E2 = form(e2)


        D1 = [[K,     C],
            [G1_4, E1]]
        D1 = petsc.assemble_matrix_block(D1)
        D1.assemble()
        

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
    
        list_D        = np.array([D1, D2, D3])
        
        return list_D


    def get_listZ(self):
        mesh         = self.mesh.mesh
        mesh_tags    = self.mesh.mesh_tags
        mesh_bc_tags = self.mesh.mesh_bc_tags
        xref         = self.mesh.xref

        entity_maps_mesh = self.mesh.entity_maps_mesh
        n                = FacetNormal(mesh) # Normal to the boundaries
        
        #deg         = self.deg
        P, Q        = self.mesh.fonction_spaces()
        dx, ds, dx1 = self.mesh.integral_mesure() 
    
        p, q = TrialFunction(P), TrialFunction(Q)
        v, u = TestFunction(P), TestFunction(Q)
        
        fx1 = Function(Q)
        fx1.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))
        
        k = inner(grad(p), grad(v)) * dx
        m = inner(p, v) * dx
        c = inner(q, v)*ds(3)
        
        dp  = inner(grad(p), n) # dp/dn = grad(p) * n
        ddp = inner(grad(dp), n) # d^2p/dn^2 = grad(dp/dn) * n = grad(grad(p) * n) * n
        g1  = inner(ddp, u)*ds(3)
        g2  = inner(p, u)*ds(3)
        g3  = inner(fx1*p, u)*ds(3)
        g4  = inner(fx1**2*p, u)*ds(3)
        e1  = inner(fx1*q, u)*dx1
        e2  = inner(q, u)*dx1
    
        list_Z       = np.array([k,      m,  c, g1,     g2,    g3, g4, e1, e2])
        return list_Z
    
    
    def import_matrix(self, freq):

        list_D = self.b2p_newVersion()
        k0 = 2*np.pi*freq/c0
        Z = list_D[0]
        for i in range(1, len(list_D)):
            Z = Z + (1j*k0)**i*list_D[i]

        ope_str = 'b2p'
        ai, aj, av = Z.getValuesCSR()
        Zsp = csr_matrix((av, aj, ai))
        savemat('Z_'+ope_str+'_ref.mat', {'Z'+ope_str:Zsp})


class B2p_modified_dp_dq(Operator):

    def __init__(self, mesh):
        '''
        Constructor of the b2p operator.
        '''
        super().__init__(mesh)
        #self.list_Z, self.list_coeff_Z = self.b2p_newVersion()
        self.list_D = self.b2p_newVersion()
        
    def b2p_newVersion(self):
        '''
        Create all the constant Form of the b2p operator on the given mesh/simulation and all the coeff of the global matrix, all the constant Form of the force block vector and the related coeff
    
        input :
            mesh_info    = List[]
            submesh_info = List[]
    
        output :
            list_Z       = np.array(Form) : List of the constant matrices of the b1p operator
            list_coeff_Z = np.array() : array of the coeff as sympy expression w.r.t 'fr' symbol
    
        '''
        mesh    = self.mesh.mesh
        submesh = self.mesh.submesh
        xref    = self.mesh.xref

        entity_maps_mesh = self.mesh.entity_maps_mesh
        n                = FacetNormal(mesh) # Normal to the boundaries
        ns               = CellNormal(submesh) # Normal to the boundaries
        
        #deg        = self.deg
        P, Q        = self.mesh.fonction_spaces()
        dx, ds, dx1 = self.mesh.integral_mesure() 
    
        p, q = TrialFunction(P), TrialFunction(Q)
        v, u = TestFunction(P), TestFunction(Q)
        
        fx1_p = Function(P)
        fx1_p.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))

        fx1_q = Function(Q)
        fx1_q.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))
        
        k = inner(grad(p), grad(v)) * dx
        m = inner(p, v) * dx
        c = inner(q, v)*ds(3)
        
        dp  = inner(grad(p), n) # dp/dn = grad(p) * n
        ddp = inner(grad(dp), n) # d^2p/dn^2 = grad(dp/dn) * n = grad(grad(p) * n) * n
        #g1  = inner(ddp, u)*ds(3)
        g2  = inner(p, u)*ds(3)
        g3  = inner(4*fx1_p*p, u)*ds(3)
        g4  = inner(2*fx1_p**2*p, u)*ds(3)

        dq  = inner(grad(q), ns) # dq/dn = grad(q) * n
        e0  = inner(dq, u)*dx1
        e1  = inner(4*fx1_q*q, u)*dx1
        e2  = inner(2*q, u)*dx1

        K = form(k)
        M = form(m)
        C = form(-c, entity_maps=entity_maps_mesh)
        
        # The two following matrices are not created, their sum is
        #G1 = form(g1, entity_maps=entity_maps_mesh)
        #G4 = form(g4, entity_maps=entity_maps_mesh)
        
        G2 = form(g2, entity_maps=entity_maps_mesh)
        G3 = form(g3, entity_maps=entity_maps_mesh)
        
        G4 = form(g4, entity_maps=entity_maps_mesh)
        
        E0_1 = form(e0 + e1)
        E1 = form(e1)
        E2 = form(e2)


        D1 = [[K,     C],
              [G4, E0_1]]
        D1 = petsc.assemble_matrix_block(D1)
        D1.assemble()
        

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
    
        list_D        = np.array([D1, D2, D3])
        
        return list_D


    def get_listZ(self):
        mesh         = self.mesh.mesh
        mesh_tags    = self.mesh.mesh_tags
        mesh_bc_tags = self.mesh.mesh_bc_tags
        xref         = self.mesh.xref

        submesh           = self.mesh.submesh
        entity_maps_mesh = self.mesh.entity_maps_mesh
        n                = FacetNormal(mesh) # Normal to the boundaries
        ns               = CellNormal(submesh) # Normal to the boundaries
        
        P, Q        = self.mesh.fonction_spaces()
        dx, ds, dx1 = self.mesh.integral_mesure() 
    
        p, q = TrialFunction(P), TrialFunction(Q)
        v, u = TestFunction(P), TestFunction(Q)
        
        fx1 = Function(Q)
        fx1.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))
        
        k = inner(grad(p), grad(v)) * dx
        m = inner(p, v) * dx
        c = inner(q, v)*ds(3)
        
        #dp  = inner(grad(p), n) # dp/dn = grad(p) * n
        #ddp = inner(grad(dp), n) # d^2p/dn^2 = grad(dp/dn) * n = grad(grad(p) * n) * n
        #g1  = inner(ddp, u)*ds(3)
        g2  = inner(p, u)*ds(3)
        g3  = inner(fx1*p, u)*ds(3)
        g4  = inner(fx1**2*p, u)*ds(3)

        dq  = inner(grad(q), ns) # dq/dn = grad(q) * n
        e0  = inner(dq, u)*dx1
        e1  = inner(fx1*q, u)*dx1
        e2  = inner(q, u)*dx1
    
        list_Z       = np.array([k, m, c, g2, g3, g4, e0, e1, e2])
        return list_Z



    def import_matrix(self, freq):

        list_D = self.b2p_newVersion()
        k0 = 2*np.pi*freq/c0
        Z = list_D[0]
        for i in range(1, len(list_D)):
            Z = Z + (1j*k0)**i*list_D[i]

        ope_str = 'b2p_modified_dp_dq'
        ai, aj, av = Z.getValuesCSR()
        Zsp = csr_matrix((av, aj, ai))
        savemat('Z_'+ope_str+'_ref.mat', {'Z'+ope_str:Zsp})

class B2p_modified_r(Operator):

    def __init__(self, mesh):
        '''
        Constructor of the b2p operator.
        '''
        super().__init__(mesh)
        #self.list_Z, self.list_coeff_Z = self.b2p()
        self.list_D = self.b2p_newVersion()

    # To edit
    def b2p_newVersion(self):
        '''
        Create all the constant Form of the b2p operator on the given mesh/simulation and all the coeff of the global matrix, all the constant Form of the force block vector and the related coeff
    
        input :
            mesh_info    = List[]
            submesh_info = List[]
    
        output :
            list_Z       = np.array(Form) : List of the constant matrices of the b1p operator
            list_coeff_Z = np.array() : array of the coeff as sympy expression w.r.t 'fr' symbol
    
        '''
        mesh    = self.mesh.mesh
        submesh = self.mesh.submesh
        xref    = self.mesh.xref

        entity_maps_mesh = self.mesh.entity_maps_mesh
        n                = FacetNormal(mesh) # Normal to the boundaries
        
        #deg        = self.deg
        P, Q        = self.mesh.fonction_spaces()
        dx, ds, dx1 = self.mesh.integral_mesure() 
    
        p, q = TrialFunction(P), TrialFunction(Q)
        v, u = TestFunction(P), TestFunction(Q)
        
        fx1_p = Function(P)
        fx1_p.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))

        fx1_q = Function(Q)
        fx1_q.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))
        
        k = inner(grad(p), grad(v)) * dx
        m = inner(p, v) * dx
        c = inner(q, v)*ds(3)
        

        dp  = inner(grad(p), n) # dp/dn = grad(p) * n
        ddp = inner(grad(dp), n) # d^2p/dn^2 = grad(dp/dn) * n = grad(grad(p) * n) * n
        g1  = inner((1/fx1_p**2)*ddp, u)*ds(3)
        g2  = inner((1/fx1_p**2)*p, u)*ds(3)
        g3  = inner((4/fx1_p)*p, u)*ds(3)
        g4  = inner(2*p, u)*ds(3)
        
        e1  = inner((4/fx1_q)*q, u)*dx1
        e2  = inner((2/fx1_q**2)*q, u)*dx1

        K = form(k)
        M = form(m)
        C = form(-c, entity_maps=entity_maps_mesh)
        
        # The two following matrices are not created, their sum is
        #G1 = form(g1, entity_maps=entity_maps_mesh)
        #G4 = form(g4, entity_maps=entity_maps_mesh)
        
        G2 = form(g2, entity_maps=entity_maps_mesh)
        G3 = form(g3, entity_maps=entity_maps_mesh)
        
        G1_4 = form(g1 + g4, entity_maps=entity_maps_mesh)
        

        E1 = form(e1)
        E2 = form(e2)


        D1 = [[K,     C],
            [G1_4, E1]]
        D1 = petsc.assemble_matrix_block(D1)
        D1.assemble()
        

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
    
        list_D        = np.array([D1, D2, D3])
        
        return list_D


    def get_listZ(self):
        mesh         = self.mesh.mesh
        mesh_tags    = self.mesh.mesh_tags
        mesh_bc_tags = self.mesh.mesh_bc_tags
        xref         = self.mesh.xref

        entity_maps_mesh = self.mesh.entity_maps_mesh
        n                = FacetNormal(mesh) # Normal to the boundaries
        
        #deg         = self.deg
        P, Q        = self.mesh.fonction_spaces()
        dx, ds, dx1 = self.mesh.integral_mesure() 
    
        p, q = TrialFunction(P), TrialFunction(Q)
        v, u = TestFunction(P), TestFunction(Q)
        
        fx1 = Function(Q)
        fx1.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))
        
        k = inner(grad(p), grad(v)) * dx
        m = inner(p, v) * dx
        c = inner(q, v)*ds(3)
        
        dp  = inner(grad(p), n) # dp/dn = grad(p) * n
        ddp = inner(grad(dp), n) # d^2p/dn^2 = grad(dp/dn) * n = grad(grad(p) * n) * n
        g1  = inner((1/fx1**2)*ddp, u)*ds(3)
        g2  = inner((1/fx1**2)*p, u)*ds(3)
        g3  = inner((1/fx1)*p, u)*ds(3)
        g4  = inner(p, u)*ds(3)
        e1  = inner((1/fx1)*q, u)*dx1
        e2  = inner((1/fx1**2)*q, u)*dx1
    
        list_Z       = np.array([k, m, c, g1, g2, g3, g4, e1, e2])
        return list_Z


    def import_matrix(self, freq):

        list_D = self.b2p_newVersion()
        k0 = 2*np.pi*freq/c0
        Z = list_D[0]
        for i in range(1, len(list_D)):
            Z = Z + (1j*k0)**i*list_D[i]

        ope_str = 'b2p_modified_r'
        ai, aj, av = Z.getValuesCSR()
        Zsp = csr_matrix((av, aj, ai))
        savemat('Z_'+ope_str+'_ref.mat', {'Z'+ope_str:Zsp})

class B2p_modified_r6(Operator):

    def __init__(self, mesh):
        '''
        Constructor of the b2p operator.
        '''
        super().__init__(mesh)
        #self.list_Z, self.list_coeff_Z = self.b2p()
        self.list_D = self.b2p_newVersion()

  
    # To edit
    def b2p_newVersion(self):
        '''
        Create all the constant Form of the b2p operator on the given mesh/simulation and all the coeff of the global matrix, all the constant Form of the force block vector and the related coeff
    
        input :
            mesh_info    = List[]
            submesh_info = List[]
    
        output :
            list_Z       = np.array(Form) : List of the constant matrices of the b1p operator
            list_coeff_Z = np.array() : array of the coeff as sympy expression w.r.t 'fr' symbol
    
        '''
        mesh    = self.mesh.mesh
        submesh = self.mesh.submesh
        xref    = self.mesh.xref

        entity_maps_mesh = self.mesh.entity_maps_mesh
        n                = FacetNormal(mesh) # Normal to the boundaries
        
        #deg        = self.deg
        P, Q        = self.mesh.fonction_spaces()
        dx, ds, dx1 = self.mesh.integral_mesure() 
    
        p, q = TrialFunction(P), TrialFunction(Q)
        v, u = TestFunction(P), TestFunction(Q)
        
        fx1_p = Function(P)
        fx1_p.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))

        fx1_q = Function(Q)
        fx1_q.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))
        
        k = inner(grad(p), grad(v)) * dx
        m = inner(p, v) * dx
        c = inner(q, v)*ds(3)
        
        dp  = inner(grad(p), n) # dp/dn = grad(p) * n
        ddp = inner(grad(dp), n) # d^2p/dn^2 = grad(dp/dn) * n = grad(grad(p) * n) * n
        g1  = inner((1/fx1_p**6)*ddp, u)*ds(3)
        g2  = inner((1/fx1_p**6)*p, u)*ds(3)
        g3  = inner(4*(1/fx1_p**5)*p, u)*ds(3)
        g4  = inner(2*(1/fx1_p**4)*p, u)*ds(3)

        e1  = inner(4*(1/fx1_q**5)*q, u)*dx1
        e2  = inner(2*(1/fx1_q**6)*q, u)*dx1

        K = form(k)
        M = form(m)
        C = form(-c, entity_maps=entity_maps_mesh)
        
        # The two following matrices are not created, their sum is
        #G1 = form(g1, entity_maps=entity_maps_mesh)
        #G4 = form(g4, entity_maps=entity_maps_mesh)
        
        G2 = form(g2, entity_maps=entity_maps_mesh)
        G3 = form(g3, entity_maps=entity_maps_mesh)
        
        G1_4 = form(g1 + g4, entity_maps=entity_maps_mesh)
        

        E1 = form(e1)
        E2 = form(e2)


        D1 = [[K,     C],
              [G1_4, E1]]
        D1 = petsc.assemble_matrix_block(D1)
        D1.assemble()
        

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
    
        list_D        = np.array([D1, D2, D3])
        
        return list_D

    def get_listZ(self):
        mesh         = self.mesh.mesh
        mesh_tags    = self.mesh.mesh_tags
        mesh_bc_tags = self.mesh.mesh_bc_tags
        xref         = self.mesh.xref

        entity_maps_mesh = self.mesh.entity_maps_mesh
        n                = FacetNormal(mesh) # Normal to the boundaries
        
        #deg         = self.deg
        P, Q        = self.mesh.fonction_spaces()
        dx, ds, dx1 = self.mesh.integral_mesure() 
    
        p, q = TrialFunction(P), TrialFunction(Q)
        v, u = TestFunction(P), TestFunction(Q)
        
        fx1 = Function(Q)
        fx1.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))
        
        k = inner(grad(p), grad(v)) * dx
        m = inner(p, v) * dx
        c = inner(q, v)*ds(3)
        
        dp  = inner(grad(p), n) # dp/dn = grad(p) * n
        ddp = inner(grad(dp), n) # d^2p/dn^2 = grad(dp/dn) * n = grad(grad(p) * n) * n
        g1  = inner((1/fx1**6)*ddp, u)*ds(3)
        g2  = inner((1/fx1**6)*p, u)*ds(3)
        g3  = inner((1/fx1**5)*p, u)*ds(3)
        g4  = inner(1/fx1**4*p, u)*ds(3)
        e1  = inner((1/fx1**5)*q, u)*dx1
        e2  = inner((1/fx1**6)*q, u)*dx1
    
        list_Z       = np.array([k,      m,  c, g1,     g2,    g3, g4, e1,    e2])
        return list_Z


    def import_matrix(self, freq):

        list_D = self.b2p_newVersion()
        k0 = 2*np.pi*freq/c0
        Z = list_D[0]
        for i in range(1, len(list_D)):
            Z = Z + (1j*k0)**i*list_D[i]

        ope_str = 'b2p_modified_r6'
        ai, aj, av = Z.getValuesCSR()
        Zsp = csr_matrix((av, aj, ai))
        savemat('Z_'+ope_str+'_ref.mat', {'Z'+ope_str:Zsp})
    
    

class B2p_modified_mixB1p(Operator):

    def __init__(self, mesh):
        '''
        Constructor of the b2p operator.
        '''
        super().__init__(mesh)
        #self.list_Z, self.list_coeff_Z = self.b2p()
        self.list_D = self.b2p_newVersion()
        
 
    def b2p_newVersion(self):
        '''
        Create all the constant Form of the b2p operator on the given mesh/simulation and all the coeff of the global matrix, all the constant Form of the force block vector and the related coeff
    
        input :
            mesh_info    = List[]
            submesh_info = List[]
    
        output :
            list_Z       = np.array(Form) : List of the constant matrices of the b1p operator
            list_coeff_Z = np.array() : array of the coeff as sympy expression w.r.t 'fr' symbol
    
        '''
        mesh    = self.mesh.mesh
        submesh = self.mesh.submesh
        xref    = self.mesh.xref

        entity_maps_mesh = self.mesh.entity_maps_mesh
        n                = FacetNormal(mesh) # Normal to the boundaries
        
        #deg        = self.deg
        P, Q        = self.mesh.fonction_spaces()
        dx, ds, dx1 = self.mesh.integral_mesure() 
    
        p, q = TrialFunction(P), TrialFunction(Q)
        v, u = TestFunction(P), TestFunction(Q)
        
        fx1_p = Function(P)
        fx1_p.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))

        fx1_q = Function(Q)
        fx1_q.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))
        
        k = inner(grad(p), grad(v)) * dx
        m = inner(p, v) * dx
        c1_1 = inner(fx1_p*p, v)*ds(3)
        c1_2 = inner(p, v)*ds(3)
        c2 = inner(-q, v)*ds(3)
        
        dp  = inner(grad(p), n) # dp/dn = grad(p) * n
        ddp = inner(grad(dp), n) # d^2p/dn^2 = grad(dp/dn) * n = grad(grad(p) * n) * n
        g1  = inner(ddp, u)*ds(3)
        g2  = inner(-p, u)*ds(3)
        g3  = inner(-2*fx1_p*p, u)*ds(3)
        g4  = inner(-2*fx1_p**2*p, u)*ds(3)

        e1  = inner(4*fx1_q*q, u)*dx1
        e2  = inner(2*q, u)*dx1

        K = form(k)
        M = form(m)
        C1_1 = form(c1_1)
        C1_2 = form(c1_2)
        C2 = form(c2, entity_maps=entity_maps_mesh)
        K_C1_1 = form(k + c1_1)

        
        # The two following matrices are not created, their sum is
        #G1 = form(g1, entity_maps=entity_maps_mesh)
        #G4 = form(g4, entity_maps=entity_maps_mesh)
        
        G2 = form(g2, entity_maps=entity_maps_mesh)
        G3 = form(g3, entity_maps=entity_maps_mesh)
        
        G1_4 = form(g1 + g4, entity_maps=entity_maps_mesh)
        

        E1 = form(e1)
        E2 = form(e2)


        D1 = [[K_C1_1,  C2],
              [G1_4,    E1]]
        D1 = petsc.assemble_matrix_block(D1)
        D1.assemble()
        

        D2_00 = form(inner(Constant(mesh, PETSc.ScalarType(0))*p, v) * dx)
        D2_01 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, v) * ds(3), entity_maps=entity_maps_mesh)
        D2 = [[C1_2,   D2_01],
              [G3,         E2]]
        D2 = petsc.assemble_matrix_block(D2)
        D2.assemble()

        D3_01 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, v) * ds(3), entity_maps=entity_maps_mesh)
        D3_11 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, u) * dx1)
        D3 = [[M,  D3_01],
            [G2, D3_11]]
        D3 = petsc.assemble_matrix_block(D3)
        D3.assemble()
    
        list_D        = np.array([D1, D2, D3])
        
        return list_D

    def get_listZ(self):
        mesh         = self.mesh.mesh
        mesh_tags    = self.mesh.mesh_tags
        mesh_bc_tags = self.mesh.mesh_bc_tags
        xref         = self.mesh.xref

        entity_maps_mesh = self.mesh.entity_maps_mesh
        n                = FacetNormal(mesh) # Normal to the boundaries
        
        #deg         = self.deg
        P, Q        = self.mesh.fonction_spaces()
        dx, ds, dx1 = self.mesh.integral_mesure() 
    
        p, q = TrialFunction(P), TrialFunction(Q)
        v, u = TestFunction(P), TestFunction(Q)
        
        fx1 = Function(Q)
        fx1.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))

        fx2 = Function(P)
        fx2.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))
        
        k = inner(grad(p), grad(v)) * dx
        m = inner(p, v) * dx
        c1_1 = inner(fx2*p, v)*ds(3)
        c1_2 = inner(p, v)*ds(3)
        c2 = inner(q, v)*ds(3)
        
        dp  = inner(grad(p), n) # dp/dn = grad(p) * n
        ddp = inner(grad(dp), n) # d^2p/dn^2 = grad(dp/dn) * n = grad(grad(p) * n) * n
        g1  = inner(ddp, u)*ds(3)
        g2  = inner(p, u)*ds(3)
        g3  = inner(fx1*p, u)*ds(3)
        g4  = inner(fx1**2*p, u)*ds(3)
        e1  = inner(fx1*q, u)*dx1
        e2  = inner(q, u)*dx1
    
        list_Z       = np.array([k,      m, c1_1, c1_2, c2, g1,    g2,     g3, g4, e1,    e2])
        return list_Z


    def import_matrix(self, freq):

        list_D = self.b2p_newVersion()
        k0 = 2*np.pi*freq/c0
        Z = list_D[0]
        for i in range(1, len(list_D)):
            Z = Z + (1j*k0)**i*list_D[i]

        ope_str = 'b2p_modified_mixB1p'
        ai, aj, av = Z.getValuesCSR()
        Zsp = csr_matrix((av, aj, ai))
        savemat('Z_'+ope_str+'_ref.mat', {'Z'+ope_str:Zsp})

    

        
class B2p_modified_IBP(Operator):

    def __init__(self, mesh):
        '''

        Constructor of the b2p operator.
        '''
        super().__init__(mesh)
        #self.list_Z, self.list_coeff_Z = self.b2p()
        self.list_D = self.b2p_newVersion()
        
 
    def b2p_newVersion(self):
        '''
        Create all the constant Form of the b2p operator on the given mesh/simulation and all the coeff of the global matrix, all the constant Form of the force block vector and the related coeff
    
        input :
            mesh_info    = List[]
            submesh_info = List[]
    
        output :
            list_Z       = np.array(Form) : List of the constant matrices of the b1p operator
            list_coeff_Z = np.array() : array of the coeff as sympy expression w.r.t 'fr' symbol
    
        '''
        mesh    = self.mesh.mesh
        submesh = self.mesh.submesh
        xref    = self.mesh.xref

        entity_maps_mesh = self.mesh.entity_maps_mesh
        n                = FacetNormal(mesh) # Normal to the boundaries
        ns               = CellNormal(submesh) # Normal to the boundaries
        
        #deg        = self.deg
        P, Q        = self.mesh.fonction_spaces()
        dx, ds, dx1 = self.mesh.integral_mesure() 
    
        p, q = TrialFunction(P), TrialFunction(Q)
        v, u = TestFunction(P), TestFunction(Q)
        
        fx1_p = Function(P)
        fx1_p.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))

        fx1_q = Function(Q)
        fx1_q.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))
        
        k = inner(grad(p), grad(v)) * dx
        m = inner(p, v) * dx
        c = inner(q, v)*ds(3)
        
        dp  = inner(grad(p), n) # dp/dn = grad(p) * n
        ddp = inner(grad(dp), n) # d^2p/dn^2 = grad(dp/dn) * n = grad(grad(p) * n) * n
        dpvec = ufl.as_vector(grad(p))
        duvec = ufl.as_vector(grad(u))


        dpvect = tangential_proj(dpvec, n)
        duvect = tangential_proj(duvec, n)

        #g1   = inner(-(dpvect[0] + dpvect[1]+ dpvect[2]), duvect[0] + duvect[1]+ duvect[2]) * ds(3)
        g1   = inner(-grad(p), grad(u)) * ds(3)
        print(g1)
        g2  = inner(2*p, u)*ds(3)
        g3  = inner(4*fx1_p*p, u)*ds(3)
        g4  = inner(2*fx1_p**2*p, u)*ds(3)

        e1  = inner(4*fx1_q*q, u)*dx1
        e2  = inner(2*q, u)*dx1

        K = form(k)
        M = form(m)
        C = form(-c, entity_maps=entity_maps_mesh)
        
        # The two following matrices are not created, their sum is
        #G1 = form(g1, entity_maps=entity_maps_mesh)
        #G4 = form(g4, entity_maps=entity_maps_mesh)
        
        G2 = form(g2, entity_maps=entity_maps_mesh)
        G3 = form(g3, entity_maps=entity_maps_mesh)
        
        G1_4 = form(g1 + g4, entity_maps=entity_maps_mesh)
        

        E1 = form(e1)
        E2 = form(e2)


        D1 = [[K,     C],
            [G1_4, E1]]
        D1 = petsc.assemble_matrix_block(D1)
        D1.assemble()
        

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
    
        list_D        = np.array([D1, D2, D3])
        
        return list_D

    def get_listZ(self):
        mesh         = self.mesh.mesh
        mesh_tags    = self.mesh.mesh_tags
        mesh_bc_tags = self.mesh.mesh_bc_tags
        xref         = self.mesh.xref

        entity_maps_mesh = self.mesh.entity_maps_mesh
        n                = FacetNormal(mesh) # Normal to the boundaries
        
        #deg         = self.deg
        P, Q        = self.mesh.fonction_spaces()
        dx, ds, dx1 = self.mesh.integral_mesure() 
    
        p, q = TrialFunction(P), TrialFunction(Q)
        v, u = TestFunction(P), TestFunction(Q)
        
        fx1 = Function(Q)
        fx1.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))
        
        k = inner(grad(p), grad(v)) * dx
        m = inner(p, v) * dx
        c = inner(q, v)*ds(3)
        
        dp  = inner(grad(p), n) # dp/dn = grad(p) * n
        ddp = inner(grad(dp), n) # d^2p/dn^2 = grad(dp/dn) * n = grad(grad(p) * n) * n
        g1  = inner(ddp, u)*ds(3)
        g2  = inner(p, u)*ds(3)
        g3  = inner(fx1*p, u)*ds(3)
        g4  = inner(fx1**2*p, u)*ds(3)
        e1  = inner(fx1*q, u)*dx1
        e2  = inner(q, u)*dx1
    
        list_Z       = np.array([k,      m,  c, g1,     g2,    g3, g4, e1, e2])
        return list_Z

    def import_matrix(self, freq):

        list_D = self.b2p_newVersion()
        k0 = 2*np.pi*freq/c0
        Z = list_D[0]
        for i in range(1, len(list_D)):
            Z = Z + (1j*k0)**i*list_D[i]

        ope_str = 'b2p_modified_IBP'
        ai, aj, av = Z.getValuesCSR()
        Zsp = csr_matrix((av, aj, ai))
        savemat('Z_'+ope_str+'_ref.mat', {'Z'+ope_str:Zsp})


class B3p(Operator):

    def __init__(self, mesh):
        '''
        Constructor of the b2p operator.
        '''
        super().__init__(mesh)
        #self.list_Z, self.list_coeff_Z = self.b3p()
        self.list_D = self.b3p_newVersion()
        
   
    def b3p_newVersion(self):
        '''
        Create all the constant Form of the b2p operator on the given mesh/simulation and all the coeff of the global matrix, all the constant Form of the force block vector and the related coeff
    
        input :
            mesh_info    = List[]
            submesh_info = List[]
    
        output :
            list_Z       = np.array(Form) : List of the constant matrices of the b1p operator
            list_coeff_Z = np.array() : array of the coeff as sympy expression w.r.t 'fr' symbol
    
        '''
        mesh         = self.mesh.mesh
        submesh      = self.mesh.submesh
        mesh_tags    = self.mesh.mesh_tags
        mesh_bc_tags = self.mesh.mesh_bc_tags
        xref         = self.mesh.xref

        entity_maps_mesh = self.mesh.entity_maps_mesh
        n                = FacetNormal(mesh) # Normal to the boundaries
        
        #deg         = self.deg
        P, Q        = self.mesh.fonction_spaces()
        dx, ds, dx1 = self.mesh.integral_mesure() 
    
        p, q = TrialFunction(P), TrialFunction(Q)
        v, u = TestFunction(P), TestFunction(Q)
        
        fx1_p = Function(P)
        fx1_p.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))

        fx1_q = Function(Q)
        fx1_q.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))
        
        k = inner(grad(p), grad(v)) * dx
        m = inner(p, v) * dx
        c = inner(q, v)*ds(3)
        
        dp   = inner(grad(p), n) # dp/dn = grad(p) * n
        ddp  = inner(grad(dp), n) # d^2p/dn^2 = grad(dp/dn) * n = grad(grad(p) * n) * n
        dddp = inner(grad(ddp), n) # d^3p/dn^3 = grad(d^2p/dn^2) * n = grad(grad(dp/dn) * n) * n = grad(grad(grad(p) * n) * n) * n
        if False:
            gradp = ufl.as_vector([p.dx(0), p.dx(1), p.dx(2)])
            dp = inner(gradp, n)
            gradgradp = ufl.as_vector([dp.dx(0), dp.dx(1), dp.dx(2)])
            ddp = inner(gradgradp, n)
            gradgradgradp = ufl.as_vector([ddp.dx(0), ddp.dx(1), ddp.dx(2)])
            dddp = inner(gradgradgradp, n)

        g1   = inner(dddp, u)*ds(3)
        #g1   = inner(p.dx(0).dx(0).dx(0), u)*ds(3)


        g2   = inner(9*fx1_p*ddp, u)*ds(3)
        g3   = inner(3*ddp, u)*ds(3)
        
        g4   = inner(6*fx1_p**3*p, u)*ds(3)
        g5   = inner(18*fx1_p**2*p, u)*ds(3)
        g6   = inner(9*fx1_p*p, u)*ds(3)
        g7   = inner(p, u)*ds(3)
        
        e1   = inner(18*fx1_q**2*q, u)*dx1
        e2   = inner(18*fx1_q*q, u)*dx1
        e3   = inner(3*q, u)*dx1
    
        K = form(k)
        M = form(m)
        C = form(-c, entity_maps=entity_maps_mesh)
        
        # The two following matrices are not created, their sum is
        #G1 = form(g1, entity_maps=entity_maps_mesh)
        #G2 = form(g2, entity_maps=entity_maps_mesh)
        #G4 = form(g4, entity_maps=entity_maps_mesh)
        #G3 = form(g3, entity_maps=entity_maps_mesh)
        
        G1_2_4 = form(g1 + g2 + g4, entity_maps=entity_maps_mesh)
        G3_5 = form(g3 + g5, entity_maps=entity_maps_mesh)
        
        G6 = form(g6, entity_maps=entity_maps_mesh)
        G7 = form(g7, entity_maps=entity_maps_mesh)
        

        E1 = form(e1)
        E2 = form(e2)
        E3 = form(e3)


        D1 = [[K,       C],
              [G1_2_4, E1]]
        D1 = petsc.assemble_matrix_block(D1)
        D1.assemble()
        

        D2_00 = form(inner(Constant(mesh, PETSc.ScalarType(0))*p, v) * dx)
        D2_01 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, v) * ds(3), entity_maps=entity_maps_mesh)
        D2 = [[D2_00,   D2_01],
              [G3_5,       E2]]
        D2 = petsc.assemble_matrix_block(D2)
        D2.assemble()

        D3_01 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, v) * ds(3), entity_maps=entity_maps_mesh)
        D3 = [[M,  D3_01],
              [G6,    E3]]
        D3 = petsc.assemble_matrix_block(D3)
        D3.assemble()

        D4_00 = form(inner(Constant(mesh, PETSc.ScalarType(0))*p, v) * dx)
        D4_01 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, v) * ds(3), entity_maps=entity_maps_mesh)
        D4_11 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, u) * dx1)
        D4 = [[D4_00,  D4_01],
              [G7,    D4_11]]
        D4 = petsc.assemble_matrix_block(D4)
        D4.assemble()
    
        list_D        = np.array([D1, D2, D3, D4])
        
        return list_D

    def get_listZ(self):
        mesh         = self.mesh.mesh
        mesh_tags    = self.mesh.mesh_tags
        mesh_bc_tags = self.mesh.mesh_bc_tags
        xref         = self.mesh.xref

        entity_maps_mesh = self.mesh.entity_maps_mesh
        n                = FacetNormal(mesh) # Normal to the boundaries
        
        #deg         = self.deg
        P, Q        = self.mesh.fonction_spaces()
        dx, ds, dx1 = self.mesh.integral_mesure() 
    
        p, q = TrialFunction(P), TrialFunction(Q)
        v, u = TestFunction(P), TestFunction(Q)
        
        fx1 = Function(Q)
        fx1.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))
        
        k = inner(grad(p), grad(v)) * dx
        m = inner(p, v) * dx
        c = inner(q, v)*ds(3)
        
        dp   = inner(grad(p), n) # dp/dn = grad(p) * n
        ddp  = inner(grad(dp), n) # d^2p/dn^2 = grad(dp/dn) * n = grad(grad(p) * n) * n
        dddp = inner(grad(ddp), n) # d^3p/dn^3 = grad(d^2p/dn^2) * n = grad(grad(dp/dn) * n) * n = grad(grad(grad(p) * n) * n) * n
        
        g1   = inner(dddp, u)*ds(3)

        g2   = inner(fx1*ddp, u)*ds(3)
        g3   = inner(ddp, u)*ds(3)
        
        g4   = inner(fx1**3*p, u)*ds(3)
        g5   = inner(fx1**2*p, u)*ds(3)
        g6   = inner(fx1*p, u)*ds(3)
        g7   = inner(p, u)*ds(3)
        
        e1   = inner(fx1**2*q, u)*dx1
        e2   = inner(fx1*q, u)*dx1
        e3   = inner(q, u)*dx1
    
        list_Z       = np.array([k,      m,  c, g1, g2,    g3, g4,     g5,       g6,        g7, e1,     e2,       e3])
        return list_Z


    def import_matrix(self, freq):

        list_D = self.b3p_newVersion()
        k0 = 2*np.pi*freq/c0
        Z = list_D[0]
        for i in range(1, len(list_D)):
            Z = Z + (1j*k0)**i*list_D[i]

        ope_str = 'b3p'
        ai, aj, av = Z.getValuesCSR()
        Zsp = csr_matrix((av, aj, ai))
        savemat('Z_'+ope_str+'_ref.mat', {'Z'+ope_str:Zsp})

class B3p_modified_r(Operator):

    def __init__(self, mesh):
        '''
        Constructor of the b2p operator.
        '''
        super().__init__(mesh)
        #self.list_Z, self.list_coeff_Z = self.b3p()
        self.list_D = self.b3p_newVersion()
        

    def b3p_newVersion(self):
        '''
        Create all the constant Form of the b2p operator on the given mesh/simulation and all the coeff of the global matrix, all the constant Form of the force block vector and the related coeff
    
        input :
            mesh_info    = List[]
            submesh_info = List[]
    
        output :
            list_Z       = np.array(Form) : List of the constant matrices of the b1p operator
            list_coeff_Z = np.array() : array of the coeff as sympy expression w.r.t 'fr' symbol
    
        '''
        mesh         = self.mesh.mesh
        submesh      = self.mesh.submesh
        mesh_tags    = self.mesh.mesh_tags
        mesh_bc_tags = self.mesh.mesh_bc_tags
        xref         = self.mesh.xref

        entity_maps_mesh = self.mesh.entity_maps_mesh
        n                = FacetNormal(mesh) # Normal to the boundaries
        
        #deg         = self.deg
        P, Q        = self.mesh.fonction_spaces()
        dx, ds, dx1 = self.mesh.integral_mesure() 
    
        p, q = TrialFunction(P), TrialFunction(Q)
        v, u = TestFunction(P), TestFunction(Q)
        
        fx1_p = Function(P)
        fx1_p.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))

        fx1_q = Function(Q)
        fx1_q.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))
        
        k = inner(grad(p), grad(v)) * dx
        m = inner(p, v) * dx
        c = inner(q, v)*ds(3)
        
        dp   = inner(grad(p), n) # dp/dn = grad(p) * n
        ddp  = inner(grad(dp), n) # d^2p/dn^2 = grad(dp/dn) * n = grad(grad(p) * n) * n
        dddp = inner(grad(ddp), n) # d^3p/dn^3 = grad(d^2p/dn^2) * n = grad(grad(dp/dn) * n) * n = grad(grad(grad(p) * n) * n) * n
        if False:
            gradp = ufl.as_vector([p.dx(0), p.dx(1), p.dx(2)])
            dp = inner(gradp, n)
            gradgradp = ufl.as_vector([dp.dx(0), dp.dx(1), dp.dx(2)])
            ddp = inner(gradgradp, n)
            gradgradgradp = ufl.as_vector([ddp.dx(0), ddp.dx(1), ddp.dx(2)])
            dddp = inner(gradgradgradp, n)

        g1   = inner((1/fx1_p**3)*dddp, u)*ds(3)
        #g1   = inner(p.dx(0).dx(0).dx(0), u)*ds(3)


        g2   = inner(9*(1/fx1_p**2)*ddp, u)*ds(3)
        g3   = inner(3(1/fx1_p**3)*ddp, u)*ds(3)
        
        g4   = inner(6*p, u)*ds(3)
        g5   = inner(18*(1/fx1_p)*p, u)*ds(3)
        g6   = inner(9*(1/fx1_p**2)*p, u)*ds(3)
        g7   = inner((1/fx1_p**3)*p, u)*ds(3)
        
        e1   = inner(18*(1/fx1_q)*q, u)*dx1
        e2   = inner(18*(1/fx1_q**2)*q, u)*dx1
        e3   = inner(3(1/fx1_q**3)*q, u)*dx1
    
        K = form(k)
        M = form(m)
        C = form(-c, entity_maps=entity_maps_mesh)
        
        # The two following matrices are not created, their sum is
        #G1 = form(g1, entity_maps=entity_maps_mesh)
        #G2 = form(g2, entity_maps=entity_maps_mesh)
        #G4 = form(g4, entity_maps=entity_maps_mesh)
        #G3 = form(g3, entity_maps=entity_maps_mesh)
        
        G1_2_4 = form(g1 + g2 + g4, entity_maps=entity_maps_mesh)
        G3_5 = form(g3 + g5, entity_maps=entity_maps_mesh)
        
        G6 = form(g6, entity_maps=entity_maps_mesh)
        G7 = form(g7, entity_maps=entity_maps_mesh)
        

        E1 = form(e1)
        E2 = form(e2)
        E3 = form(e3)


        D1 = [[K,       C],
              [G1_2_4, E1]]
        D1 = petsc.assemble_matrix_block(D1)
        D1.assemble()
        

        D2_00 = form(inner(Constant(mesh, PETSc.ScalarType(0))*p, v) * dx)
        D2_01 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, v) * ds(3), entity_maps=entity_maps_mesh)
        D2 = [[D2_00,   D2_01],
              [G3_5,       E2]]
        D2 = petsc.assemble_matrix_block(D2)
        D2.assemble()

        D3_01 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, v) * ds(3), entity_maps=entity_maps_mesh)
        D3 = [[M,  D3_01],
              [G6,    E3]]
        D3 = petsc.assemble_matrix_block(D3)
        D3.assemble()

        D4_00 = form(inner(Constant(mesh, PETSc.ScalarType(0))*p, v) * dx)
        D4_01 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, v) * ds(3), entity_maps=entity_maps_mesh)
        D4_11 = form(inner(Constant(submesh, PETSc.ScalarType(0))*q, u) * dx1)
        D4 = [[D4_00,  D4_01],
              [G7,    D4_11]]
        D4 = petsc.assemble_matrix_block(D4)
        D4.assemble()
    
        list_D        = np.array([D1, D2, D3, D4])
        
        return list_D

    def get_listZ(self):
        mesh         = self.mesh.mesh
        mesh_tags    = self.mesh.mesh_tags
        mesh_bc_tags = self.mesh.mesh_bc_tags
        xref         = self.mesh.xref

        entity_maps_mesh = self.mesh.entity_maps_mesh
        n                = FacetNormal(mesh) # Normal to the boundaries
        
        #deg         = self.deg
        P, Q        = self.mesh.fonction_spaces()
        dx, ds, dx1 = self.mesh.integral_mesure() 
    
        p, q = TrialFunction(P), TrialFunction(Q)
        v, u = TestFunction(P), TestFunction(Q)
        
        fx1 = Function(Q)
        fx1.interpolate(lambda x: 1/np.sqrt((x[0]-xref[0])**2 + (x[1]-xref[1])**2 + (x[2]-xref[2])**2))
        
        k = inner(grad(p), grad(v)) * dx
        m = inner(p, v) * dx
        c = inner(q, v)*ds(3)
        
        dp   = inner(grad(p), n) # dp/dn = grad(p) * n
        ddp  = inner(grad(dp), n) # d^2p/dn^2 = grad(dp/dn) * n = grad(grad(p) * n) * n
        dddp = inner(grad(ddp), n) # d^3p/dn^3 = grad(d^2p/dn^2) * n = grad(grad(dp/dn) * n) * n = grad(grad(grad(p) * n) * n) * n
        
        g1   = inner((1/fx1**3)*dddp, u)*ds(3)
        #g1   = inner((1/fx1**6)*dddp, u)*ds(3)

        g2   = inner((1/fx1**2)*ddp, u)*ds(3)
        g3   = inner((1/fx1**3)*ddp, u)*ds(3)
        #g2   = inner((1/fx1**5)*ddp, u)*ds(3)
        #g3   = inner((1/fx1**6)*ddp, u)*ds(3)
        
        g4   = inner(p, u)*ds(3)
        g5   = inner((1/fx1)*p, u)*ds(3)
        g6   = inner((1/fx1**2)*p, u)*ds(3)
        g7   = inner((1/fx1**3)*p, u)*ds(3)
        #g4   = inner((1/fx1**3)*p, u)*ds(3)
        #g5   = inner((1/fx1**4)*p, u)*ds(3)
        #g6   = inner((1/fx1**5)*p, u)*ds(3)
        #g7   = inner((1/fx1**6)*p, u)*ds(3)
        
        e1   = inner((1/fx1)*q, u)*dx1
        e2   = inner((1/fx1**2)*q, u)*dx1
        e3   = inner((1/fx1**3)*q, u)*dx1
        #e1   = inner((1/fx1**4)*q, u)*dx1
        #e2   = inner((1/fx1**5)*q, u)*dx1
        #e3   = inner((1/fx1**6)*q, u)*dx1
    
        list_Z       = np.array([k,      m,  c, g1, g2,    g3, g4,     g5,       g6,        g7, e1,     e2,       e3])
        return list_Z


    def import_matrix(self, freq):

        list_D = self.b3p_newVersion()
        k0 = 2*np.pi*freq/c0
        Z = list_D[0]
        for i in range(1, len(list_D)):
            Z = Z + (1j*k0)**i*list_D[i]

        ope_str = 'b3p_modified_r'
        ai, aj, av = Z.getValuesCSR()
        Zsp = csr_matrix((av, aj, ai))
        savemat('Z_'+ope_str+'_ref.mat', {'Z'+ope_str:Zsp})



class Loading:
    '''
    A Loading is a force applied on the mesh. So far this class only implement the vibrating plate, but this class could be implemented as the abstract class Operator
    '''

    def __init__(self, mesh):
        '''
        Constructor of the class Loading
        '''
        self.mesh                      = mesh
        #self.list_F, self.list_coeff_F = self.f()
        self.F = self.f_newVersion()


    # To edit
    def f_newVersion(self):
        '''
        This function create the coefficients list of the vibration plate.
        input :

        output :
            list_F       = np.array() : list of the frequency constant vectors
            list_coeff_F = np.array() : list of the coeffient
        '''
        
        submesh      = self.mesh.submesh
    
        P, Q        = self.mesh.fonction_spaces()
        _, ds, dx1 = self.mesh.integral_mesure() 
    
        v, u = TestFunction(P), TestFunction(Q)
        
        f = form(inner(1, v) * ds(1))
        zero = form(inner(Constant(submesh, PETSc.ScalarType(0)), u) * dx1)

        F = [f, zero]
        F = petsc.assemble_vector_nest(F)
    
        return F




def sub_matrix(Q, start, end):
    '''
    This function is to obtain the sub matrix need for the correction term (P_q_w)
    intput :
        Q     = PETScMatType : the matrix where the norms and the scalar products are stored
        start = int : start index, reality index
        end   = int : end index, reality index

    output : 
        submatrix = np.array() : sub matrix, as a numpy matrix, because the size will remain low
    '''
     
    row_is    = PETSc.IS().createStride(end  - start + 1, first=start - 1, step=1)
    col_is    = PETSc.IS().createStride(end - start + 1, first=start - 1, step=1)
    submatrix = Q.createSubMatrix(row_is, col_is)

    row_is.destroy()
    col_is.destroy()

    submatrix = submatrix.getValues([i for i in range(end - start+1)], [i for i in range(end - start+1)])
    return submatrix
        
def P_Q_w(Q, alpha, beta, omega):
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
    
    P_q = np.identity(alpha - beta) #create the identity matrix M*M with M = alpha - beta

    for t in range(omega, beta+1):
        sub_Q = sub_matrix(Q, t, alpha - beta + t - 1)
        sub_Q = np.linalg.inv(sub_Q)
        P_q   = np.dot(P_q, sub_Q)

    # The following lignes convert the result to a PETSc type
    P_q_w = PETSc.Mat().create()
    P_q_w.setSizes(P_q.shape, P_q.shape)
    P_q_w.setType("seqdense")  
    P_q_w.setFromOptions()
    P_q_w.setUp()

    for i in range(P_q.shape[0]):
        P_q_w.setValues(i, [j for j in range(P_q.shape[1])], P_q[i], PETSc.InsertMode.INSERT_VALUES)   
    P_q_w.assemble()
    return P_q_w

def Zn_Fn_matrices(Z, F, Vn):
    '''
    This perform a reduction of the matrix and the vector through a given basis
    input :
        Z  = PETScMatType : assembled global matrix
        F  = PETScVecType : assembled force vector
        Vn = PETScMatType : orthonomalized basis

    output :
        Zn  = PETScMatType : reduced global matrix
        Fn  = PETScVecType : reduced force vector
        
    '''

    Vn_T = Vn.duplicate()
    Vn.copy(Vn_T)
    Vn_T.hermitianTranspose()
    Vn_T.assemble()

    Zn = PETSc.Mat().create()
    Zn.setSizes(Vn.getSizes()[1])  
    Zn.setType("seqdense")  
    Zn.setFromOptions()
    Zn.setUp()

    C = PETSc.Mat().create()
    C.setSizes(Vn.getSize()) 
    C.setType("seqdense")
    C.setFromOptions()
    C.setUp()

    Z.matMult(Vn, C) # C = Z * Vn
    Vn_T.matMult(C, Zn) # Zn = Vn_T * C = Vn_T * Z * Vn
    C.destroy()

    Fn = Zn.createVecLeft()
    Vn_T.mult(F, Fn) # Fn = Vn_T * F


    return Zn, Fn

def listDr_matrices(list_D, F, Vn) :
    list_Dr = []
    # Compute the reduced matrices
    Vn_T = Vn.duplicate()
    Vn.copy(Vn_T)
    Vn_T.hermitianTranspose()
    Vn_T.assemble()
    print(f'Vn size : {Vn.getSize()}')
    print(f'Vn_T size : {Vn_T.getSize()}')

    for Di in list_D:
        Dir = Vn_T.matMult(Di) 
        Dir = Dir.matMult(Vn)
        Dir.assemble()
        list_Dr.append(Dir)

    Fr = list_Dr[0].createVecLeft()
    Vn_T.mult(F, Fr) # Fn = Vn_T * F
    
    return list_Dr, Fr


def SVD_ortho(Vn):
    '''
    This function performs an orthogonalization of a basis with a singular value decomposition, with the power of SLEPc.
    input :
        Vn = PETScMatType : initial basis

    output :
        L = PETScMatType : result basis
    '''
    n   = Vn.getSize()[0]
    m   = Vn.getSize()[1]
    print(f'n : {n}')
    
    svd = SLEPc.SVD().create()
    svd.setOperator(Vn)
    #svd.setFromOptions()
    svd.setDimensions(m)
    svd.solve()
    
    nsv = svd.getConverged()
    print(f'svd : {nsv}')
    
    L = PETSc.Mat().createDense([n, nsv], comm=PETSc.COMM_WORLD)
    L.setUp()  
    
    for i in range(nsv):
        
        u = PETSc.Vec().createSeq(n)
        v = PETSc.Vec().createSeq(m)

        sigma = svd.getSingularTriplet(i, u, v)
        #print(sigma)

        for j in range(n):
            L[j, i] = u[j]
    
    L.assemble()

    return L

def get_cond_nb(Z):
    # Creating the SVD solver
    svd = SLEPc.SVD().create()
    svd.setOperator(Z)

    svd.setDimensions(nsv=1000)
    svd.setFromOptions()

    # Solving the singular value decomposition
    svd.solve()

    # Recovery of extreme singular values
    list_sigma = []
    sigma_max = svd.getValue(0)
    sigma_min = None

    for i in range(svd.getConverged()):
        sigma = svd.getValue(i)
        list_sigma.append(sigma)
        #print(f'sigma :{sigma}')
        if sigma > 0 and (sigma_min is None or sigma < sigma_min):
            sigma_min = sigma

    if sigma_min is None or sigma_min == 0:
        raise RuntimeError("The smallest singular value is zero, the number of conditions is infinite.")

    condition_number = sigma_max / sigma_min
    #print(f'Conditioning number :{condition_number}')
    return condition_number, list_sigma

def get_cond_nbV2(Z):
    t1 = time()
    Z_fct1 = Z.copy()
    Z_fct1.convert("seqdense")
    Z_fct = Z_fct1.getDenseArray()
    #print(Z_fct)

    S = la.eigvals(Z_fct)
    t2 = time()
    print(f"scipy done in {t2-t1}")
    #condition_number = max(S) / min(S)
    #S = la.eigvals(Z_fct)

    # Créer un objet EPS pour résoudre le problème aux valeurs propres
    eps = SLEPc.EPS().create()
    # Assigner la matrice A au solveur
    eps.setOperators(Z)
    # Définir les options (par exemple, calculer 10 valeurs propres)
    eps.setDimensions(nev=len(S))  # nev : nombre de valeurs propres à calculer
    # Définir le type de problème aux valeurs propres (standard)
    #eps.setProblemType(SLEPc.EPS.ProblemType.HEP)  # HEP : Hermitian Eigenvalue Problem
    # Choisir le solveur (par exemple, Krylov-Schur, par défaut)
    eps.setFromOptions()
    # Résoudre le problème
    eps.solve()
    
    # Obtenir le nombre de valeurs propres convergées
    nconv = eps.getConverged()
    
    # Initialiser une liste pour stocker les valeurs propres
    eigenvalues = []
    
    # Récupérer les valeurs propres convergées
    for i in range(nconv):
        eigenvalue = eps.getEigenvalue(i)
        eigenvalues.append(eigenvalue)
    t3 = time()
    print(f"SLEPc done in {t3 -t2}")
    return S, np.array(eigenvalues)

def get_cond_nbV3(A):
    """
    Calcule le nombre de conditionnement d'une matrice PETSc (A) 
    à partir de ses valeurs singulières via SLEPc.
    
    Parameters
    ----------
    A : petsc4py.PETSc.Mat
        Matrice dont on veut le conditionnement.

    Returns
    -------
    cond : float
        Nombre de conditionnement calculé comme sigma_max / sigma_min.
    
    Raises
    ------
    ValueError:
        Si la plus petite valeur singulière est nulle ou si la résolution 
        du problème SVD échoue.
    """
    # Créer un objet SVD
    svd = SLEPc.SVD().create()
    svd.setOperator(A)
    # On veut au moins la plus grande et la plus petite valeur singulière
    svd.setDimensions(2, PETSc.DECIDE, PETSc.DECIDE)
    # On peut choisir un type de résolution, par exemple: svd.setType(SLEPc.SVD.Type.LAPACK)
    # SVD par défaut : (en général LANCZOS ou TRLAN selon la config)
    svd.setFromOptions()

    # Résoudre le SVD
    svd.solve()
    
    nconv = svd.getConverged()
    if nconv < 2:
        raise ValueError("Pas assez de valeurs singulières convergées pour calculer le conditionnement.")

    # Préallouer les vecteurs u et v pour la valeur singulière max
    u_max = A.createVecLeft()   # Vecteur pour u
    v_max = A.createVecRight()  # Vecteur pour v
    sigma_max = svd.getSingularTriplet(0, u_max, v_max)

    # Préallouer les vecteurs u et v pour la valeur singulière min
    u_min = A.createVecLeft()
    v_min = A.createVecRight()
    sigma_min = svd.getSingularTriplet(nconv - 1, u_min, v_min)

    if sigma_min == 0.0:
        raise ValueError("La plus petite valeur singulière est nulle, conditionnement infini.")

    cond = sigma_max / sigma_min
    return cond
    
def row_norms(A):
    
    # Obtain the dimensions of the matrix
    m, n = A.getSize()

    # Initialise an array to store row norms
    list_row_norms = []

    # Calculate the norm of each line
    for i in range(m):
        row = A.getRow(i)[1]  # Retrieves non-zero values from the line
        row_norm = np.linalg.norm(row)
        list_row_norms.append(row_norm)
    return list_row_norms

def column_norms(A):

    # Obtain the dimensions of the matrix
    m, n = A.getSize()

    # Initialise an array to store columns norms
    list_column_norms = []

    # Calculate the norm of each columns
    for j in range(n):
        column_values = np.zeros(m, dtype = 'complex')  # Initialise a complete array the size of the column

        # Browse each row to retrieve the values in column j
        for i in range(m):
            column_values[i] = A.getValue(i, j)

        # Calculate the norm of the column
        column_norm = np.linalg.norm(column_values)
        list_column_norms.append(column_norm)
    
    return list_column_norms

def SVD_ortho2(Vn):
    '''
    This function might be delete further. This function performs an orthogonalization of a basis with a singular value decomposition based on python computation. It converts the PETScMatType to a numpy array, does the computation, and gives back the orthogonalized matrix in the PETScMatType type.
    input :
        Vn = PETScMatType : initial basis

    output :
        V_petsc = PETScMatType : result basis
    '''
    Vn = Vn.getDenseArray()

    L, S, R = la.svd(Vn)
    
    print(f'len S : {len(S)}')
    print(S)
    L_star = L[:,0:len(S)]
    print(L_star.shape)
    V_n = L_star

    V_petsc = PETSc.Mat().create()
    V_petsc.setSizes((V_n.shape[0], V_n.shape[1]))
    V_petsc.setType('aij')  
    V_petsc.setUp()

    for i in range(V_n.shape[0]):
        for j in range(V_n.shape[1]):
            V_petsc[i, j] = V_n[i, j]
    V_petsc.assemble()

    return V_petsc

def check_ortho(Vn):
    '''
    This function plot the scalar product between 2 following vector inside a basis, to check if they are orthogonal one to each other.
    input :
        Vn = PETScMatType : the basis

    output :
    '''
    N = Vn.getSize()[1]
    for i in range(N-1):
        vec1 = Vn.getColumnVector(i)
        vec2 = Vn.getColumnVector(i+1)
        result = vec1.dot(vec2)
        print("vec"+str(i)+" . vec"+str(i+1)+" = "+str(result))

def get_wcawe_param():
    with open("wcawe_param.txt", 'r') as file:
        lines = file.readlines()
    
    sections = ["Dir", "Geometry", "Case", "Operator", "Lc", "DimP", "DimQ"]
    sections_val = []
    # Variable pour indiquer si la ligne précédente contient "section"
    nb_sec = 0
    previous_line_is_section = False

    # Liste pour stocker les nouvelles lignes
    updated_lines = []
    
    for line in lines:
        if previous_line_is_section:
            sections_val.append(line)
            previous_line_is_section = False
        else:
            updated_lines.append(line)
            if any(section in line for section in sections):
                previous_line_is_section = True
                nb_sec +=1
    
    dir  = sections_val[0].removesuffix("\n")
    geo  = sections_val[1].removesuffix("\n")
    case = sections_val[2].removesuffix("\n")
    ope  = sections_val[3].removesuffix("\n")
    lc   = float(sections_val[4])
    dimP = int(sections_val[5])
    dimQ = int(sections_val[6])
    return dir, geo, case, ope, lc, dimP, dimQ
    
def parse_wcawe_param():
    frequencies = []
    n_values = []

    with open("wcawe_param.txt", 'r') as file:
        lines = file.readlines()

        freq_section = False
        n_section = False

        for line in lines:

            if line.startswith('%') or '\n' == line:
                continue

            if "List frequencies" in line:
                freq_section = True
                n_section = False
                continue
            elif "List N" in line:
                freq_section = False
                n_section = True
                continue

            if freq_section:
                frequencies.append(int(line.strip()))
            elif n_section:
                n_values.append(int(line.strip()))

    return frequencies, n_values

def store_results(s, freqvec, Pav):
    with open('FOM_data/'+s+'.txt', 'w') as fichier:    
        for i in range(len(freqvec)):        
            fichier.write('{}\t{}\n'.format(freqvec[i], Pav[i]))

def store_resultsv2(list_s, freqvec, Pav, simu):
    z_center = simu.compute_radiation_factor(freqvec, Pav)
    dict_s = {
        "geo"  : list_s[0],
        "case" : list_s[1],
        "ope"  : list_s[2],
        "lc"   : list_s[3],
        "dimP" : list_s[4],
        "dimQ" : list_s[5]
    }
    s = "classical_"
    for key, value in dict_s.items():
        s+= value
        if key != "dimQ":
            s+= "_"
    s+=".txt"
    print(s)
    with open("classical/"+s, 'w') as fichier:    
        for i in range(len(freqvec)):        
            fichier.write('{}\t{}\n'.format(freqvec[i], z_center[i]))

def store_resultsv3(s, freqvec, Pav, simu):
    z_center = simu.compute_radiation_factor(freqvec, Pav)
    with open('./FOM_data/'+s+'.txt', 'w') as fichier:    
        for i in range(len(freqvec)):   

            fichier.write('{}\t{}\n'.format(freqvec[i], z_center[i]))

def store_resultsv4(s, freqvec, Pav, simu):
    z_center = simu.compute_radiation_factor(freqvec, Pav)
    chemin_actuel = Path(__file__).resolve().parent
    chemin_fichier = chemin_actuel / "FOM_data"/ f"{s}.txt"
    with open(chemin_fichier, 'w') as fichier:    
        for i in range(len(freqvec)):   

            fichier.write('{}\t{}\n'.format(freqvec[i], z_center[i]))

def store_results_wcawe(list_s, freqvec, Pav, simu, file_wcawe_para_list):
    z_center = simu.compute_radiation_factor(freqvec, Pav)
    dict_s = {
        "geo"  : list_s[0],
        "case" : list_s[1],
        "ope"  : list_s[2],
        "lc"   : list_s[3],
        "dimP" : list_s[4],
        "dimQ" : list_s[5]
    }
    s_dir = ""
    for key, value in dict_s.items():
        s_dir+= value
        if key != "dimQ":
            s_dir+= "_"
    print(s_dir)
    if "modified_r" in list_s[2]:
        directory_path = "wcawe/classical/modified_r/" + s_dir
    else:
        directory_path = "wcawe/classical/" + s_dir
    # Create the directory if it doesn't exist
    os.makedirs(directory_path, exist_ok=True)
    if file_wcawe_para_list[0]:
        list_freq, list_N = parse_wcawe_param()
    else:
        list_freq, list_N = file_wcawe_para_list[1], file_wcawe_para_list[2]
    s = ""
    for freq in list_freq:
        s+=str(freq)
        s+="Hz_"
    for i in range(len(list_N)):
        s+=str(list_N[i])
        if i !=len(list_N)-1:
            s+="_"
        else:
            s+=".txt"
    print(s)
    file_path = os.path.join(directory_path, s)
    with open(file_path, 'w') as fichier:    
        for i in range(len(freqvec)):        
            fichier.write('{}\t{}\n'.format(freqvec[i], z_center[i]))


def store_results_wcawev2(list_s, list_freq, list_N, freqvec, Pav, simu):
    z_center = simu.compute_radiation_factor(freqvec, Pav)
    dict_s = {
        "geo"  : list_s[0],
        "case" : list_s[1],
        "ope"  : list_s[2],
        "lc"   : list_s[3],
        "dimP" : list_s[4],
        "dimQ" : list_s[5]
    }
    s_dir = ""
    for key, value in dict_s.items():
        s_dir+= value
        if key != "dimQ":
            s_dir+= "_"
    print(s_dir)

    s = ""
    for freq in list_freq:
        s+=str(freq)
        s+="Hz_"
    for i in range(len(list_N)):
        s+=str(list_N[i])
        if i !=len(list_N)-1:
            s+="_"
        else:
            s+=".txt"
    print(s)

    chemin_actuel = Path(__file__).resolve().parent
    if "modified" in list_s[2]:
        chemin_dir = chemin_actuel / "wcawe"/ "modified_ope" / s_dir
    else:
        chemin_dir = chemin_actuel / "wcawe"/ "classical" / s_dir
    os.makedirs(chemin_dir, exist_ok=True)
    chemin_fichier = chemin_dir / f"{s}"
    
    with open(chemin_fichier, 'w') as fichier:    
        for i in range(len(freqvec)):   

            fichier.write('{}\t{}\n'.format(freqvec[i], z_center[i]))

def import_frequency_sweep(s):
    with open('FOM_data/'+s+".txt", "r") as f:
        freqvec = list()
        Pav     = list()
        for line in f:
            if "%" in line:
                continue
            data    = line.split()
            freqvec.append(data[0])
            Pav.append(data[1])
            freqvec = [float(element) for element in freqvec]
            Pav     = [complex(element) for element in Pav]
    
    freqvec = np.array(freqvec)
    Pav     = np.array(Pav)
    
    return freqvec, Pav

def import_frequency_sweepv2(s):
    with open('classical/'+s+".txt", "r") as f:
        freqvec = list()
        Pav     = list()
        for line in f:
            if "%" in line:
                continue
            data    = line.split()
            freqvec.append(data[0])
            Pav.append(data[1])
            freqvec = [float(element) for element in freqvec]
            Pav     = [complex(element) for element in Pav]
    
    freqvec = np.array(freqvec)
    Pav     = np.array(Pav)
    
    return freqvec, Pav

def compute_analytical_radiation_factor(freqvec, radius):
    k_output = 2*np.pi*freqvec/c0
    Z_analytical = (1-2*special.jv(1,2*k_output*radius)/(2*k_output*radius) + 1j*2*special.struve(1,2*k_output*radius)/(2*k_output*radius)) #The impedance is divided by rho * c0, it becames the radiation coefficient
    return Z_analytical

def plot_analytical_result_sigma(ax, freqvec, radius):
    Z_analytical = compute_analytical_radiation_factor(freqvec, radius)
    ax.plot(freqvec, Z_analytical.real, label = r'$\sigma_{ana}$', c = 'blue')
    ax.legend()

def import_COMSOL_result(s):
    s = "COMSOL_data/"+s+"_COMSOL_results.txt"
    with open(s, "r") as f:
        frequency = list()
        results = list()
        for line in f:
            if "%" in line:
                # on saute la ligne
                continue
            data = line.split()
            frequency.append(data[0])
            results.append(data[1])
            frequency = [float(element) for element in frequency]
            results = [float(element) for element in results]
    return frequency, results

def least_square_err(freqvec1, Z_center1, freqvec2, Z_center2):
    
    if len(freqvec1) >=  len(freqvec2):
        check = set(freqvec2) <= set(freqvec1)
        freqvec_in = freqvec2
        freqvec_out = freqvec1
        Pav_in = Z_center2
        Pav_out = Z_center1
    else :
        check = set(freqvec1) <= set(freqvec2)
        freqvec_in = freqvec1
        freqvec_out = freqvec2
        Pav_in = Z_center1
        Pav_out = Z_center2
    if check :
        
        err = 0
        for i in range(len(freqvec_in)):
            err += (Pav_in[i] - Pav_out[np.where(freqvec_out == freqvec_in[i])[0][0]])**2
    
        return err
    else :
        print("something went wrong")
        return None
    
def harry_plotter(space, sol, str_value, show_edges = True):

    u_topology, u_cell_types, u_geometry = plot.vtk_mesh(space)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_grid.point_data[str_value] = sol.x.array.real
    u_grid.set_active_scalars(str_value)
    u_plotter = pyvista.Plotter()
    u_plotter.add_mesh(u_grid, show_edges=show_edges)
    u_plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        u_plotter.show()

def harry_plotterv2(spaces, sols, str_values, show_edges = True):

    nb_graphs = len(spaces)

    if nb_graphs % 3 == 0 :
        rows = nb_graphs // 3
    else :
        rows = nb_graphs // 3 + 1 
    
    if nb_graphs <= 3 :
        cols = nb_graphs
    else : 
        cols = 3
    
    ii = 0
    plotter = pyvista.Plotter(shape=(rows, cols))

    for i in range(rows):
        for j in range(cols):
            space     = spaces[ii]
            sol       = sols[ii]
            str_value = str_values[ii]
            
            plotter.subplot(i, j)
            u_topology, u_cell_types, u_geometry = plot.vtk_mesh(space)
            u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
            u_grid.point_data[str_value] = sol.x.array.real
            u_grid.set_active_scalars(str_value)
            plotter.add_mesh(u_grid, show_edges=show_edges)
            plotter.view_xy()
            ii += 1
    
    if not pyvista.OFF_SCREEN:
        plotter.show()

def invert_petsc_matrix(A: PETSc.Mat) -> PETSc.Mat:
    """
    Calcule l'inverse d'une matrice PETSc donnée.

    Paramètres
    ----------
    A : PETSc.Mat
        Matrice PETSc à inverser.

    Retour
    ------
    A_inv : PETSc.Mat
        Matrice inverse de A (sous forme PETSc.Mat).
    """
    # Vérifier que la matrice est carrée
    m, n = A.getSize()
    if m != n:
        raise ValueError("La matrice doit être carrée pour être inversée.")

    # Créer une matrice pour stocker l'inverse
    A_inv = PETSc.Mat().create()
    A_inv.setSizes(A.getSizes())
    A_inv.setType(A.getType())
    A_inv.setUp()

    # Créer un solveur linéaire PETSc
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setType("gmres")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")


    # Résolution colonne par colonne pour calculer l'inverse
    for i in range(n):
        #print(i)
        # Créer un vecteur unité (e_i) de taille n
        e_i = PETSc.Vec().createSeq(n)
        e_i.setValue(i, 1.0)
        e_i.assemble()

        # Résoudre A * x = e_i
        x = A.createVecRight()  # Vecteur de résultat
        ksp.solve(e_i, x)

        # Ajouter x en tant que i-ème colonne de A_inv
        col_values = x.getArray()
        for j in range(n):
            A_inv.setValue(j, i, col_values[j])

    # Assembler la matrice inverse
    A_inv.assemble()

    return A_inv

def soar(A, B, u, n):
    """
    SOAR procedure to generate a reduced basis.
    
    Args:
        A (PETSc.Mat): Input matrix A.
        B (PETSc.Mat): Input matrix B.
        u (PETSc.Vec): Initial vector.
        n (int): Number of basis vectors to compute.
    
    Returns:
        np.ndarray: Matrix of shape (m, n) containing the basis vectors.
    """
    # Ensure vectors are not destroyed
    A.assemble()
    B.setUp()
    #u_array = u.getArray()
    #print("Solution u (premiers éléments) :", u_array[:10])

    
    # Normalize initial vector
    q = u.copy()
    q_norm = q.norm(PETSc.NormType.NORM_2)
    q.scale(1.0 / q_norm)
    
    # Basis storage (PETSc Vecs)
    Q = [q]
    T = []  # Hessenberg matrix columns
    
    # Initial f vector
    m = u.size
    f = PETSc.Vec().createMPI(m, comm=u.comm)
    f.set(0.0)
    
    for j in range(n-1):  # Generate n-1 additional vectors
        # Step 4: r = A*q_j + B*f
        r = Q[j].copy()
        temp = Q[j].copy()
        #print(f' r before A.mult(Q[j], r) : {r.view()}')
        #print(f' A before A.mult(Q[j], r) : {A.view()}')
        A.mult(Q[j], r)
        #print(f' r after A.mult(Q[j], r) : {r.view()}')
        B.mult(f, temp)
        r.axpy(1.0, temp)  # r = A*q_j + B*f
        #A.mult(temp, r) # r = A*q_j
        #r = r + temp
        
        
        
        # Orthogonalize against previous vectors
        h = []
        for i in range(j+1):    
            # Step 5-8: Classical Gram-Schmidt
            h_i = r.dot(Q[i])
            h.append(h_i)
            r.axpy(-h_i, Q[i])  # r = r - h_i*Q[i]
            #print(f' Q[i] = Q[{i}] before Gram-Schmidt : {Q[i].view()}')
            #print(f' r before Gram-Schmidt : {r.view()}')
            #r  = r - r.dot(Q[i]) * Q[i]
        
        # Step 9: Compute norm
        #print(f'r : {r.view()}')
        h_j1 = r.norm()
        #print(f'h_j1 : {h_j1}')
        h.append(h_j1)
        #print(f'h : {h}')
        T.append(h.copy())  # Store column j
        #print(f'T : {T}')
        
        # Step 10-18: Handle breakdown and update basis
        if h_j1 > 1e-12:
            q_new = r.copy()
            q_new.scale(1.0 / h_j1)
            Q.append(q_new)
        else:
            print("breakdown")
            # Handle breakdown (set q_{j+1} to zero)
            q_new = PETSc.Vec().createMPI(m, comm=u.comm)
            q_new.set(0.0 + 1j*0.0)
            Q.append(q_new)
        
        # Step 12/16: Compute f = Q_j * inv(T_hat) * e_j
        # Construct T_hat (lower triangular)
        j_current = j + 1  # Size of T_hat is (j+1) x (j+1)
        T_hat = np.zeros((j_current, j_current), dtype=np.complex128)
        
        for col in range(j_current):
            if col >= len(T):
                break
            t_col = T[col]
            for row in range(col+1):
                if row+1 < len(t_col):
                    T_hat[row, col] = t_col[row+1]
        
        # Solve T_hat x = e_j (last canonical vector)
        e = np.zeros(j_current)
        e[j] = 1.0 if j < j_current else 0.0  # e_j in current dimension
        #print(f'T_hat :\n{T_hat}')
        x = np.linalg.solve(T_hat, e)
        
        # Update f = Q_0...j * x
        f.set(0.0)
        for i in range(j_current):
            f.axpy(x[i], Q[i])
    
    # Collect basis vectors (convert to numpy array)
    basis = np.zeros((m, n), dtype=np.complex128)
    ### Create the empty basis
    Vn = PETSc.Mat().create()
    Vn.setSizes((m, n))  
    Vn.setType("seqdense")  
    Vn.setFromOptions()
    Vn.setUp()    
    for i in range(n):
        #print(f"norm of q{i} : {np.linalg.norm(Q[i].getArray())}")
        basis[:, i] = Q[i].getArray()
        Vn.setValues([ii for ii in range(m)], i, Q[i].getArray(), PETSc.InsertMode.INSERT_VALUES) #Vn[0] = v1
    
    Vn.assemble()
    #print(f'Vn : {Vn.view()}')
    return basis, Vn

def tangential_proj(u, n):
    #print(f"u : {u}")
    #print(f"u size: {u.ufl_shape[0]}")
    proj_u = (ufl.Identity(n.ufl_shape[0]) - ufl.outer(n, n)) * u
    #print(f"proj_u : {proj_u}")
    return proj_u