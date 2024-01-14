import torch

def Solve_Piezoelectricity_3D_a(C0:torch.Tensor, K0:torch.Tensor , e0:torch.Tensor,
                                 eP:torch.Tensor, G_elas:torch.Tensor, G_piezo:torch.Tensor, G_dielec:torch.Tensor,
                                 eps0:torch.Tensor, eps_ext:torch.Tensor, E_ext:torch.Tensor,
                                 Nx:int, Ny:int, Nz:int, P:torch.Tensor):

    """
    Solves the constitutive and balance equations using FFT.

    C0, K0, e0 - homogeneous material's stiffness, piezoelectric and dielectric tesnors

    eP - heterogeneuos piezoelectric tensor

    G_elas, G_piezo, G_dielec - Calculateed Green's functions in Fourier space

    eps0 - sponteneous strain

    eps_ext - applied external strain

    E_ext - applied electric field

    Nx, Ny, Nz - number of grid points in each direction

    P - sponteneous polarization
    """

    # Initialize and change the dtype in Fourier spaec
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
    
    niter=100
    for itr in range(niter):
        print(itr)
        eps_elas = eps_tot+eps_ext-eps0
        E_tot = E + E_ext
        sigma = torch.einsum('ijkl,klxyz->ijxyz',C0, eps_elas) - torch.einsum('kijxyz,kxyz->ijxyz',eP,E_tot)
        D = torch.einsum('ijkxyz,jkxyz->ixyz',eP, eps_elas) + torch.einsum('ij,jxyz->ixyz',K0, E_tot) + P
        sig_norm_new = torch.norm(sigma.ravel(), p=2).item()
        D_norm_new = torch.norm(D.ravel(), p=2).item()
        err_s = abs((sig_norm_new - sig_norm) / sig_norm)
        err_d = abs((D_norm_new - D_norm )/ D_norm)
        print('err_s: ',err_s)
        print('err_d: ',err_d)
        if max(err_s,err_d) < 1e-4:
            break
        D_norm = D_norm_new
        sig_norm = sig_norm_new
        tau = sigma - torch.einsum('ijkl,klxyz->ijxyz',C0, eps_elas) + torch.einsum('kij,kxyz->ijxyz',e0,E_tot)
        rho = D - torch.einsum('ijk,jkxyz->ixyz',e0, eps_elas) - torch.einsum('ij,jxyz->ixyz',K0, E_tot) - P
        tau_fft = fft(tau, dim=(2,3,4))
        rho_fft = fft(rho, dim=(1,2,3))
        C0_complex = C0.to(torch.complex64)
        alpha = tau_fft-torch.einsum('ijkl,klxyz->ijxyz',C0_complex,eps0_fft)
        beta = rho_fft + P_fft - torch.einsum('ijk,jkxyz->ixyz',e0_complex,eps0_fft)
        eps_tot_fft = -torch.einsum('ijklxyz,klxyz->ijxyz',G_elas_complex,alpha) - torch.einsum('kijxyz,kxyz->ijxyz',G_piezo_complex, beta)
        eps_tot_fft[:,:,0,0]=0
        E_fft = torch.einsum('ijkxyz,jkxyz->ixyz',G_piezo_complex, alpha) + torch.einsum('ijxyz,jxyz->ixyz',G_dielec_complex, beta)
        E_fft[:,0,0]=0
        eps_tot = ifft(eps_tot_fft, dim=(2,3,4)).real
        E = ifft(E_fft, dim=(1,2,3)).real
    del eps_tot, E, P_fft, eps0_fft, tau, rho
    del tau_fft, rho_fft, alpha, beta, E_fft, eps_tot_fft

    return sigma,D,eps_elas,E_tot
