import torch
from torch.fft import fftn as fft, ifftn as ifft

import math
import sys

def Solve_Piezoelectricity(C0:torch.Tensor, K0:torch.Tensor , e0:torch.Tensor, eP:torch.Tensor,
                                G_elas:torch.Tensor, G_piezo:torch.Tensor, G_dielec:torch.Tensor,
                                eps0:torch.Tensor, eps_ext:torch.Tensor, E_ext:torch.Tensor,
                                Nx:int, Ny:int, Nz:int, P:torch.Tensor,
                                number_interations = 100, tol = 1e-4):

    """
    Solves the constitutive and balance equations using FFT.
    Returns stress, electric displacement, elastic strain and total electric field

    C0, K0, e0 - homogeneous material's stiffness, piezoelectric and dielectric tesnors

    eP - heterogeneuos piezoelectric tensor

    G_elas, G_piezo, G_dielec - Calculateed Green's functions in Fourier space

    eps0 - sponteneous strain

    eps_ext - applied external strain

    E_ext - applied electric field

    Nx, Ny, Nz - number of grid points in each direction

    P - sponteneous polarization

    number_interations - number of interations, default is 100

    tol - given tolerance, default is 1e-4
    """

    # Initialize and change the dtype in Fourier space
    C0_complex = C0.to(torch.complex64)
    e0_complex = e0.to(torch.complex64)
    G_elas_complex = G_elas.to(torch.complex64)
    G_piezo_complex = G_piezo.to(torch.complex64)
    G_dielec_complex = G_dielec.to(torch.complex64)

    # Initialize the norms
    sig_norm = 1e-8
    D_norm = 1e-8

    # Initialize the total strain and electric field
    eps_tot = torch.zeros_like(eps0, dtype=torch.float32)
    E = torch.zeros_like(E_ext, dtype=torch.float32)

    # Apply FFT for the sponteneous polarization and strain
    P_fft = fft(P, dim=(1,2,3))
    eps0_fft = fft(eps0, dim=(2,3,4))

    # start the loop
    for itr in range(number_interations):

        # elastic strain
        eps_elas = eps_tot+eps_ext-eps0

        # total electric field: depolarization + applied
        E_tot = E + E_ext

        # constitutive equations, sigma & D
        sigma = torch.einsum('ijkl, klxyz -> ijxyz', C0, eps_elas) - torch.einsum('kijxyz, kxyz -> ijxyz', eP, E_tot)
        D = torch.einsum('ijkxyz, jkxyz -> ixyz', eP, eps_elas) + torch.einsum('ij, jxyz -> ixyz', K0, E_tot) + P

        # calculate the norms
        sig_norm_new = torch.norm(sigma.ravel(), p=2).item()
        D_norm_new = torch.norm(D.ravel(), p=2).item()

        # find the error
        err_s = abs((sig_norm_new - sig_norm) / sig_norm)
        err_d = abs((D_norm_new - D_norm )/ D_norm)

        # print iteration number and errors
        print(f"Iteration: {itr} | S. error = {err_s:.2E} | D. error = {err_d:.2E}")
        # print(f"Iteration: {itr} | S. error = {err_s} | D. error = {err_d}")

        # check for nan and inf errors and break the loop if the error is below 1e-4
        if math.isnan(err_s) is True or math.isinf(err_s) is True:
            print(f"Error: Iteration loop terminated due to the presence of NaN or Inf ")
            sys.exit()
        elif max(err_s,err_d) < tol:
            break

        # update the norms
        D_norm = D_norm_new
        sig_norm = sig_norm_new

        # determine the heterogeneuos stress tesnsor and elec. displacement vector
        tau = sigma - torch.einsum('ijkl, klxyz -> ijxyz', C0, eps_elas) + torch.einsum('kij, kxyz -> ijxyz', e0, E_tot)
        rho = D - torch.einsum('ijk, jkxyz -> ixyz', e0, eps_elas) - torch.einsum('ij, jxyz -> ixyz', K0, E_tot) - P

        # do FFT
        tau_fft = fft(tau, dim=(2,3,4))
        rho_fft = fft(rho, dim=(1,2,3))

        # get alpha and beta
        alpha = tau_fft-torch.einsum('ijkl, klxyz -> ijxyz', C0_complex, eps0_fft)
        beta = rho_fft + P_fft - torch.einsum('ijk, jkxyz -> ixyz', e0_complex, eps0_fft)

        # Calculate the total strain in Fourier space
        eps_tot_fft = -torch.einsum('ijklxyz, klxyz -> ijxyz', G_elas_complex, alpha) - torch.einsum('kijxyz, kxyz -> ijxyz', G_piezo_complex, beta)
        eps_tot_fft[:,:,0,0] = 0

        # Calculate the total electric field in Fourier space
        E_fft = torch.einsum('ijkxyz, jkxyz -> ixyz', G_piezo_complex, alpha) + torch.einsum('ijxyz, jxyz -> ixyz', G_dielec_complex, beta)
        E_fft[:,0,0] = 0

        # In real space
        eps_tot = ifft(eps_tot_fft, dim=(2,3,4)).real
        E = ifft(E_fft, dim=(1,2,3)).real

    # delete unused tensors to save more memory
    del eps_tot, E, P_fft, eps0_fft, tau, rho
    del tau_fft, rho_fft, alpha, beta, E_fft, eps_tot_fft

    # return stress, electric displacement, elastic strain and total electric field
    return sigma, D, eps_elas, E_tot
