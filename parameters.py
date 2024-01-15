"""
This file contains all the data necessary for the phase-field simulations

Material data:
    - store all material parameters here
    - function Mat_Data_BTO_MD() contains material parameters for BTO

Initial polarization can also be imported from here for different domain types:
    - random -> polarization vector field starts with an initial
                uniform distribution over [-0.1, +0.1]
    - 90 -> polarization vector field starts with three 90° sharp interfaces
            Note, here must be Nx=Ny
    - 180 -> polarization vector field starts with a single 180° sharp interface
"""

import numpy as np
import itertools
import torch
import sys


def Landau_Polynomial_Coeffs():
    """
    Returns the Landau polynomial coefficients.
    """
    a1, a2, a3, a4 = -0.1, -2.8, 0.4, 1.9

    return a1, a2, a3, a4

# Functions to convert Voigt to full tensors
def full_3x3_to_Voigt_6_index_3D(i, j):
    if i == j:
        return i
    return 6-i-j

def Voigt_to_full_Tensor_3D(C):
    C = np.asarray(C)
    C_out = np.zeros((3,3,3,3), dtype=np.float32)
    for i, j, k, l in itertools.product(range(3), range(3), range(3), range(3)):
        Voigt_i = full_3x3_to_Voigt_6_index_3D(i, j)
        Voigt_j = full_3x3_to_Voigt_6_index_3D(k, l)
        C_out[i, j, k, l] = C[Voigt_i, Voigt_j]
    return C_out

# Material tensors
def BTO_MD():

    """
    Returns all the materials parameters obtained from MD simulations
    """
    # stiffness tensor components
    C11, C12, C44 = 201e9, 164e9, 138e9  # J/m³

    C_cub = np.array([[C11, C12, C12, 0, 0, 0],
                      [C12, C11, C12, 0, 0, 0],
                      [C12, C12, C11, 0, 0, 0],
                      [0,   0,   0, C44, 0, 0],
                      [0,   0,   0, 0, C44, 0],
                      [0,   0,   0, 0, 0, C44]])

    # Make a full tensor
    C0 = torch.tensor(Voigt_to_full_Tensor_3D(C_cub))

    # Piezoelectric tensor components
    e31, e33, e15 = -0.49, 4.5, 18.6   # C/m²

    # dielectric tensor
    k = 130 * 8.85418782e-12
    K0 = torch.eye(3) * k # F/m

    # max remanent strain
    a, c = 4, 4.044
    e0 = 2*(c - a)/(c + 2*a)

    # max remanent polarization
    P0 = 0.127   # C/m²

    # 180° DW width
    l_180 = 0.6e-9

    # energy scaling parameter
    G = 7.55E6  # J/m³

    # mobility
    mob = 618

    # grad coeff
    mu = 0.5

    # return all of them
    return C0, K0, P0, e0, e31, e33, e15, G, l_180, mu, mob

def Landau_Polynomial_Coeffs_Schrade():
    """
    Returns the Landau polynomial coefficients.
    """
    a1      = -86/75
    a2      = -53/75
    a3      = 134/25
    a4      = 64/75

    return a1, a2, a3, a4

def BTO_3D_Schrade():

    G       = 12e-3                       # J/m²=N/m Interfacial Energy
    l_wall  = 1.5E-9                      # m: Thickness of interface: Length Scale
    P0      = 0.26                        # C/m²: Maximum polarization
    mob     = 26/75*1e3                   # A/Vm: mobility beta^-1
    c_tet   = 4.032                         # angstrom
    a_tet   = 3.992                         # angstrom
    e00     = 2*(c_tet-a_tet)/(c_tet+2*a_tet)

     # ----------------------- MATERIAL PARAMETERS ----------------------------
    k_sep   = 0.70    # Separation coefficient
    k_grad  = 0.35   # Gradient Energy coefficient

    # piezoelectric tensor components
    e31, e33, e15 = (-0.7, 6.7, 34.2)  # e33 = 6.7
    # e_ver = e33; e_per = e31; e_hor = e15

    # e_ver = e_piezo[0]; e_per = e_piezo[1]; e_hor = e_piezo[2]
    # e33 = e_piezo[0]; e31 = e_piezo[1]; e15 = e_piezo[2]

    # Elastic tensor comps
    C11 = 22.2e10
    C12 = 11.1e10
    C44 = 6.1e10

    C_cub = np.array([[C11, C12, C12, 0, 0, 0],
                      [C12, C11, C12, 0, 0, 0],
                      [C12, C12, C11, 0, 0, 0],
                      [0,   0,   0, C44, 0, 0],
                      [0,   0,   0, 0, C44, 0],
                      [0,   0,   0, 0, 0, C44]])

    C0 = torch.tensor(Voigt_to_full_Tensor_3D(C_cub))

    k = 19.5e-9

    K0 = torch.eye(3)*k
    return G, l_wall, P0, mob, e00, k_sep, k_grad, e31, e33, e15, C0, K0

# Inital polarization
def initial_polarization(Nx, Ny, domain_type, Nz):

    torch.manual_seed(51254)
    print("Initial polarization for 3D simulations")

    if domain_type == 'random':

        # generates random initial polarization
        Px = (0.1 * (2.0 * torch.rand(Nx, Ny, Nz) - 1.0))#.requires_grad_()
        Py = (0.1 * (2.0 * torch.rand(Nx, Ny, Nz) - 1.0))#.requires_grad_()
        Pz = (0.1 * (2.0 * torch.rand(Nx, Ny, Nz) - 1.0))#.requires_grad_()

    elif domain_type == '180':

        # generates a single 180° DW
        print("180° domain type")
        Px = torch.zeros((Nx,Ny,Nz))
        Py = torch.zeros((Nx,Ny,Nz))
        Pz = torch.ones((Nx,Ny,Nz))
        Pz[:,int(Ny/2):,:]=-1

    elif domain_type == '90':

        # generates 90° DW structure
        from scipy import linalg
        first_row = torch.zeros(Nz)
        nn=Nz/2
        first_row[:int(nn)]=1
        first_row[2*int(nn)-1:3*int(nn)]=1

        first_col = torch.ones(Ny)
        first_col[:int(nn)]=0
        first_col[2*int(nn)-1:3*int(nn)]=0

        Pyy = linalg.toeplitz(first_col, first_row)
        Pxx = (1-Pyy)*-1
        # create 90° DW in yz plane
        Px = torch.zeros((Nx,Ny,Nz))
        Py = torch.zeros((Nx,Ny,Nz))
        Py[:] = Pxx
        Pz = torch.zeros((Nx,Ny,Nz))
        Pz[:] = Pyy

        print(Px.shape, Py.shape, Pz.shape)

    elif domain_type == 'minus_z':
        Px = torch.zeros((Nx,Ny,Nz))
        Py = torch.zeros((Nx,Ny,Nz))
        Pz = -torch.ones((Nx,Ny,Nz))

    elif domain_type == 'plus_z':
        Px = torchp.zeros((Nx,Ny,Nz))
        Py = torch.zeros((Nx,Ny,Nz))
        Pz = torch.ones((Nx,Ny,Nz))

    return torch.stack((Px, Py, Pz))
