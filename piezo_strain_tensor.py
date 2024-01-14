import torch
import itertools

def Piezo_Strain_Tensor(P:torch.Tensor, e33:float, e31:float, e15:float, e00:float,
                        device:torch.device):

    """
    Calculates the piezoelectric and spontaneous strain tensors
    Returns the piezoelectric tesnor, homogeneous piezoelectric tensor, sponteneous strain tensor

    P - spontaneous polarization

    e33, e31, e15 - piezoelectric tensor components in ferroelectric phase

    e00 - maximum remanent strain

    device - the device, "cpu" or "cuda"
    """

    # Get the square root of the polarization norm
    norm_sq = P[0]**2 + P[1]**2 + P[2]**2     # norm of the polarizatioin vector P

    # Unit tensor
    I = torch.eye(3).to(device)

    # Initialize the piezoelectric tensor
    eP = torch.zeros((3, 3, 3, *P[0].shape)).to(device)         # P dependen piez. tensor
    for k, i, j in itertools.product(range(3), repeat=3):
        eP[k, i, j] = (P[i] * P[j] * P[k] * (e33 - e31 - e15) + norm_sq * (e31 * I[i, j] * P[k]
                           + e15/2 * (I[k, i] * P[j] + I[k, j] * P[i])))

    # homogeneous piez. tensor, volume average of e(P)
    e0 = torch.zeros((3, 3, 3)).to(device)
    for i, j, k in itertools.product(range(3), repeat=3):
        e0[i, j, k] = torch.mean(eP[i, j, k])

    # spontaneous strain, depends on P
    eps0 = torch.zeros((3, 3, *P[0].shape)).to(device)
    for i, j in itertools.product(range(3), repeat=2):
        eps0[i, j] = 1.5 * e00 * (P[i] * P[j] - 1/3 * norm_sq * I[i, j])

    # return piezoelectric tesnor, homogeneous piezoelectric tensor, sponteneous strain tensor
    return eP, e0, eps0


def Piezo_Strain_Tensor_Derivative(P, e33, e31, e15, e00, device):

    """
    Calculates the derivative of the piezoelectric tesnor and sponteneous strain tensor
    wrt polarization P.

    Returns the derivatives

    P - spontaneous polarization

    e33, e31, e15 - piezoelectric tensor components in ferroelectric phase

    e00 - maximum remanent strain

    device - the device, "cpu" or "cuda"
    """

    # --------------- Derivaive of the piezoelectric tensor wrt P --------------
    norm_sq = P[0]**2 + P[1]**2 + P[2]**2     # norm of the polarizatioin vector P

    deP_dPx = torch.zeros((3,3,3,*P[0].shape)).to(device)        # derivative wrt P[0]
    deP_dPy = torch.zeros((3,3,3,*P[0].shape)).to(device)       # derivative wrt P[1]
    deP_dPz = torch.zeros((3,3,3,*P[0].shape)).to(device)        # derivative wrt P[2]

    em = -e15 - e31 + e33

    deP_dPx[0,0,0] = 3*P[0]**2*em  + 2*P[0]*(P[0]*e15 + P[0]*e31) + (e15 + e31)*norm_sq
    deP_dPy[0,0,0] = 2*P[1]*(P[0]*e15 + P[0]*e31)
    deP_dPz[0,0,0] = 2*P[2]*(P[0]*e15 + P[0]*e31)

    deP_dPx[0,0,1] = P[0]*P[1]*e15 + 2*P[0]*P[1]*em
    deP_dPy[0,0,1] = P[0]**2*em  + P[1]**2*e15 + e15*norm_sq/2
    deP_dPz[0,0,1] = P[1]*P[2]*e15

    deP_dPx[0,0,2] = P[0]*P[2]*e15 + 2*P[0]*P[2]*em
    deP_dPy[0,0,2] = P[1]*P[2]*e15
    deP_dPz[0,0,2] = P[0]**2*em  + P[2]**2*e15 + e15*norm_sq/2

    deP_dPx[0,1,0] = deP_dPx[0,0,1]#P[0]*P[1]*e15 + 2*P[0]*P[1]*em
    deP_dPy[0,1,0] = deP_dPy[0,0,1]#P[0]**2*em  + P[1]**2*e15 + e15*norm_sq/2
    deP_dPz[0,1,0] = deP_dPz[0,0,1]#P[1]*P[2]*e15

    deP_dPx[0,1,1] = 2*P[0]**2*e31 + P[1]**2*em  + e31*norm_sq
    deP_dPy[0,1,1] = 2*P[0]*P[1]*e31 + 2*P[0]*P[1]*em
    deP_dPz[0,1,1] = 2*P[0]*P[2]*e31

    deP_dPx[0,1,2] = P[1]*P[2]*em
    deP_dPy[0,1,2] = P[0]*P[2]*em
    deP_dPz[0,1,2] = P[0]*P[1]*em

    deP_dPx[0,2,0] = deP_dPx[0,0,2]#P[0]*P[2]*e15 + 2*P[0]*P[2]*em
    deP_dPy[0,2,0] = deP_dPy[0,0,2]#P[1]*P[2]*e15
    deP_dPz[0,2,0] = deP_dPz[0,0,2]#P[0]**2*em  + P[2]**2*e15 + e15*norm_sq/2

    deP_dPx[0,2,1] = deP_dPx[0,1,2]#P[1]*P[2]*em
    deP_dPy[0,2,1] = deP_dPy[0,1,2]#P[0]*P[2]*em
    deP_dPz[0,2,1] = deP_dPz[0,1,2]#P[0]*P[1]*em

    deP_dPx[0,2,2] = 2*P[0]**2*e31 + P[2]**2*em  + e31*norm_sq
    deP_dPy[0,2,2] = 2*P[0]*P[1]*e31
    deP_dPz[0,2,2] = 2*P[0]*P[2]*e31 + 2*P[0]*P[2]*em

    deP_dPx[1,0,0] = 2*P[0]*P[1]*e31 + 2*P[0]*P[1]*em
    deP_dPy[1,0,0] = P[0]**2*em  + 2*P[1]**2*e31 + e31*norm_sq
    deP_dPz[1,0,0] = 2*P[1]*P[2]*e31

    deP_dPx[1,0,1] = P[0]**2*e15 + P[1]**2*em  + e15*norm_sq/2
    deP_dPy[1,0,1] = P[0]*P[1]*e15 + 2*P[0]*P[1]*em
    deP_dPz[1,0,1] = P[0]*P[2]*e15

    deP_dPx[1,0,2] = P[1]*P[2]*em
    deP_dPy[1,0,2] = P[0]*P[2]*em
    deP_dPz[1,0,2] = P[0]*P[1]*em

    deP_dPx[1,1,0] = deP_dPx[1,0,1]#P[0]**2*e15 + P[1]**2*em  + e15*norm_sq/2
    deP_dPy[1,1,0] = deP_dPy[1,0,1]#P[0]*P[1]*e15 + 2*P[0]*P[1]*em
    deP_dPz[1,1,0] = deP_dPz[1,0,1]#P[0]*P[2]*e15

    deP_dPx[1,1,1] = 2*P[0]*(P[1]*e15 + P[1]*e31)
    deP_dPy[1,1,1] = 3*P[1]**2*em  + 2*P[1]*(P[1]*e15 + P[1]*e31) + (e15 + e31)*norm_sq
    deP_dPz[1,1,1] = 2*P[2]*(P[1]*e15 + P[1]*e31)

    deP_dPx[1,1,2] = P[0]*P[2]*e15
    deP_dPy[1,1,2] = P[1]*P[2]*e15 + 2*P[1]*P[2]*em
    deP_dPz[1,1,2] = P[1]**2*em  + P[2]**2*e15 + e15*norm_sq/2

    deP_dPx[1,2,0] = deP_dPx[1,0,2]#P[1]*P[2]*em
    deP_dPy[1,2,0] = deP_dPy[1,0,2]#P[0]*P[2]*em
    deP_dPz[1,2,0] = deP_dPz[1,0,2]#P[0]*P[1]*em

    deP_dPx[1,2,1] = deP_dPx[1,1,2]#P[0]*P[2]*e15
    deP_dPy[1,2,1] = deP_dPy[1,1,2]#P[1]*P[2]*e15 + 2*P[1]*P[2]*em
    deP_dPz[1,2,1] = deP_dPz[1,1,2]#P[1]**2*em  + P[2]**2*e15 + e15*norm_sq/2

    deP_dPx[1,2,2] = 2*P[0]*P[1]*e31
    deP_dPy[1,2,2] = 2*P[1]**2*e31 + P[2]**2*em  + e31*norm_sq
    deP_dPz[1,2,2] = 2*P[1]*P[2]*e31 + 2*P[1]*P[2]*em

    deP_dPx[2,0,0] = 2*P[0]*P[2]*e31 + 2*P[0]*P[2]*em
    deP_dPy[2,0,0] = 2*P[1]*P[2]*e31
    deP_dPz[2,0,0] = P[0]**2*em  + 2*P[2]**2*e31 + e31*norm_sq

    deP_dPx[2,0,1] = P[1]*P[2]*em
    deP_dPy[2,0,1] = P[0]*P[2]*em
    deP_dPz[2,0,1] = P[0]*P[1]*em

    deP_dPx[2,0,2] = P[0]**2*e15 + P[2]**2*em  + e15*norm_sq/2
    deP_dPy[2,0,2] = P[0]*P[1]*e15
    deP_dPz[2,0,2] = P[0]*P[2]*e15 + 2*P[0]*P[2]*em

    deP_dPx[2,1,0] = deP_dPx[2,0,1]#P[1]*P[2]*em
    deP_dPy[2,1,0] = deP_dPy[2,0,1]#P[0]*P[2]*em
    deP_dPz[2,1,0] = deP_dPz[2,0,1]#P[0]*P[1]*em

    deP_dPx[2,1,1] = 2*P[0]*P[2]*e31
    deP_dPy[2,1,1] = 2*P[1]*P[2]*e31 + 2*P[1]*P[2]*em
    deP_dPz[2,1,1] = P[1]**2*em  + 2*P[2]**2*e31 + e31*norm_sq

    deP_dPx[2,1,2] = P[0]*P[1]*e15
    deP_dPy[2,1,2] = P[1]**2*e15 + P[2]**2*em  + e15*norm_sq/2
    deP_dPz[2,1,2] = P[1]*P[2]*e15 + 2*P[1]*P[2]*em

    deP_dPx[2,2,0] = deP_dPx[2,0,2]#P[0]**2*e15 + P[2]**2*em  + e15*norm_sq/2
    deP_dPy[2,2,0] = deP_dPy[2,0,2]#P[0]*P[1]*e15
    deP_dPz[2,2,0] = deP_dPz[2,0,2]#P[0]*P[2]*e15 + 2*P[0]*P[2]*em

    deP_dPx[2,2,1] = deP_dPx[2,1,2]#P[0]*P[1]*e15
    deP_dPy[2,2,1] = deP_dPy[2,1,2]#P[1]**2*e15 + P[2]**2*em  + e15*norm_sq/2
    deP_dPz[2,2,1] = deP_dPz[2,1,2]#P[1]*P[2]*e15 + 2*P[1]*P[2]*em

    deP_dPx[2,2,2] = 2*P[0]*(P[2]*e15 + P[2]*e31)
    deP_dPy[2,2,2] = 2*P[1]*(P[2]*e15 + P[2]*e31)
    deP_dPz[2,2,2] = 3*P[2]**2*em  + 2*P[2]*(P[2]*e15 + P[2]*e31) + (e15 + e31)*norm_sq

    # --------------- Derivaive of the spon. strain tensor wrt P ---------------
    deps0_P_dPx = torch.zeros((3,3,*P[0].shape)).to(device)          # Derivaive of spont. strain wrt P[1]
    deps0_P_dPy = torch.zeros((3,3,*P[0].shape)).to(device)           # Derivaive of spont. strain wrt P[2]
    deps0_P_dPz = torch.zeros((3,3,*P[0].shape)).to(device)           # Derivaive of spont. strain wrt P[2]

    deps0_P_dPx[0,0] = 2.0*P[0]*e00
    deps0_P_dPy[0,0] = -1.0*P[1]*e00
    deps0_P_dPz[0,0] = -1.0*P[2]*e00

    deps0_P_dPx[0,1] = 1.5*P[1]*e00
    deps0_P_dPy[0,1] = 1.5*P[0]*e00
    deps0_P_dPz[0,1] = 0

    deps0_P_dPx[0,2] = 1.5*P[2]*e00
    deps0_P_dPy[0,2] = 0
    deps0_P_dPz[0,2] = 1.5*P[0]*e00

    deps0_P_dPx[1,0] = deps0_P_dPx[0,1]
    deps0_P_dPy[1,0] = deps0_P_dPy[0,1]
    deps0_P_dPz[1,0] = deps0_P_dPz[0,1]

    deps0_P_dPx[1,1] = -1.0*P[0]*e00
    deps0_P_dPy[1,1] = 2.0*P[1]*e00
    deps0_P_dPz[1,1] = -1.0*P[2]*e00

    deps0_P_dPx[1,2] = 0
    deps0_P_dPy[1,2] = 1.5*P[2]*e00
    deps0_P_dPz[1,2] = 1.5*P[1]*e00

    deps0_P_dPx[2,0] = deps0_P_dPx[0,2]
    deps0_P_dPy[2,0] = deps0_P_dPy[0,2]
    deps0_P_dPz[2,0] = deps0_P_dPz[0,2]

    deps0_P_dPx[2,1] = deps0_P_dPx[1,2]
    deps0_P_dPy[2,1] = deps0_P_dPy[1,2]
    deps0_P_dPz[2,1] = deps0_P_dPz[1,2]

    deps0_P_dPx[2,2] = -1.0*P[0]*e00
    deps0_P_dPy[2,2] = -1.0*P[1]*e00
    deps0_P_dPz[2,2] = 2.0*P[2]*e00

    # --------------- Derivaive of the elastic strain tensor wrt P -------------
    deps_elas_dPx = -deps0_P_dPx
    deps_elas_dPy = -deps0_P_dPy
    deps_elas_dPz = -deps0_P_dPz

    # delete unused tensors
    del deps0_P_dPx, deps0_P_dPy, deps0_P_dPz, norm_sq

    # return derivatives
    return deP_dPx, deP_dPy, deP_dPz, deps_elas_dPx, deps_elas_dPy, deps_elas_dPz
