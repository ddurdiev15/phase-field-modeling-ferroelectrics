import torch

def Bulk_Energy(C0:torch.Tensor, eps_elas:torch.Tensor,
                    eP:torch.Tensor, E:torch.Tensor, K:torch.Tensor, P:torch.Tensor):

    """
    Calculates the bulk energy = elastic + piezoelectric + electric energies.
    Reeturn elastic, piezoelectric, electric energies

    C0 - homogeneous stiffness tensor

    eps_elas - elastic strain

    eP - piezoelectric tesnor

    E - total electric FIELD

    K - dielectric tesnor

    P - spontaneous polarization
    """

    # Elastic energy
    H_elas = 0.5 * np.einsum('ijkl, ijxyz, klxyz -> xyz', C0, eps_elas, eps_elas)

    # Piezoelectric energy
    H_piezo = -np.einsum('ijkxyz, ijxyz, kxyz -> xyz', eP, eps_elas, E)

    # Electric energy
    H_elec = -np.einsum('ij,ixyz,jxyz->xyz', K, E, E) - np.einsum('ixyz,ixyz->xyz',P,E)

    # return those energies
    return H_elas, H_piezo, H_elec

def Bulk_Energy_Derivative(C0:torch.Tensor, eps_elas:torch.Tensor, eP:torch.Tensor,
                            E:torch.Tensor, deP_dPx:torch.Tensor, deP_dPy:torch.Tensor,
                            deP_dPz:torch.Tensor, deps_elas_dPx:torch.Tensor,
                            deps_elas_dPy:torch.Tensor, deps_elas_dPz:torch.Tensor):
    """
    Calculates the derivatives of the energies.
    Returns the derivative of the elastic, piezoelectric and electric energies wrt to polarization.

    C0 - homogeneous stiffness tensor

    eps_elas - elastic strain

    eP - piezoelectric tesnor

    E - total electric FIELD

    deP_dPx, deP_dPy, deP_dPz - derivatives piezoelectric tensor wrt Px, Py, Pz

    deps_elas_dPx, deps_elas_dPy, deps_elas_dPz - derivatives of sponteneous strain tensor wrt Px, Py, Pz

    """
    # Derivaive of the elastic energy
    dH_elas_dPx = 0.5*(torch.einsum('ijkl, ijxyz, klxyz -> xyz', C0, deps_elas_dPx, eps_elas) +
                       torch.einsum('ijkl, ijxyz, klxyz -> xyz', C0, eps_elas, deps_elas_dPx) )
    dH_elas_dPy = 0.5*(torch.einsum('ijkl, ijxyz, klxyz -> xyz', C0, deps_elas_dPy, eps_elas) +
                       torch.einsum('ijkl, ijxyz, klxyz -> xyz', C0, eps_elas, deps_elas_dPy) )
    dH_elas_dPz = 0.5*(torch.einsum('ijkl, ijxyz, klxyz -> xyz', C0, deps_elas_dPz, eps_elas) +
                       torch.einsum('ijkl, ijxyz, klxyz -> xyz', C0, eps_elas, deps_elas_dPz) )

    dH_elas_dP = torch.stack((dH_elas_dPx, dH_elas_dPy, dH_elas_dPz))

    del dH_elas_dPx, dH_elas_dPy, dH_elas_dPz

    # Derivaive of the piezoelectric energy
    dH_piezo_dPx = -(torch.einsum('ijkxyz, ijxyz, kxyz -> xyz', deP_dPx, eps_elas, E) +
                       torch.einsum('ijkxyz, ijxyz, kxyz -> xyz', eP, deps_elas_dPx, E) )
    dH_piezo_dPy = -(torch.einsum('ijkxyz, ijxyz, kxyz -> xyz', deP_dPy, eps_elas, E) +
                       torch.einsum('ijkxyz, ijxyz, kxyz  -> xyz', eP, deps_elas_dPy, E) )
    dH_piezo_dPz = -(torch.einsum('ijkxyz, ijxyz, kxyz -> xyz', deP_dPz, eps_elas, E) +
                       torch.einsum('ijkxyz, ijxyz, kxyz -> xyz', eP, deps_elas_dPz, E) )

    dH_piezo_dP = torch.stack((dH_piezo_dPx, dH_piezo_dPy, dH_piezo_dPz))

    del dH_piezo_dPx, dH_piezo_dPy, dH_piezo_dPz

    # derivative of the electric energy
    dH_elec_dP = -E

    # Derivative of the bulk energy
    dH_bulk_dP = dH_elas_dP + dH_piezo_dP + dH_elec_dP

    del dH_elas_dP, dH_piezo_dP, dH_elec_dP

    return dH_bulk_dP

def Landau_Polynomial_Derivative(P, a1, a2, a3, a4):
    
    """
    Calculates the derivative of the Landau polynomial wrt P

    P - spontaneous polarization

    a1, a2, a3, a4 - polynomial coefficients
    """

    dPsi_dPx = a1 * 2*P[0] + a2 * 4*P[0]**3 + a3 * 2*P[0] * (P[1]**2 + P[2]**2) + a4 * 6*P[0]**5
    dPsi_dPy = a1 * 2*P[1] + a2 * 4*P[1]**3 + a3 * 2*P[1] * (P[0]**2 + P[2]**2) + a4 * 6*P[1]**5
    dPsi_dPz = a1 * 2*P[2] + a2 * 4*P[2]**3 + a3 * 2*P[2] * (P[0]**2 + P[1]**2) + a4 * 6*P[2]**5

    # combine them
    dPsi_dP = torch.stack((dPsi_dPx, dPsi_dPy, dPsi_dPz))

    del dPsi_dPx, dPsi_dPy, dPsi_dPz

    return dPsi_dP
