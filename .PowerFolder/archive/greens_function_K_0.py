import torch

def Green_Operator_Piezoelectric_3D_a(C0:torch.Tensor, e0:torch.Tensor, K0:torch.Tensor,
                                            n:torch.Tensor, Nx:int, Ny:int, Nz:int, device:torch.device):

    """
    Calculates the Green's functions in Fourier spaces

    C0, e0, K0 - homogeneous material's stiffness, piezoelectric and dielectric tesnors

    n - Fourier frequencies

    Nx, Ny, Nz - number of grid points in each direction

    device - torch device; "cpu" or "cuda"
    """

    A_c = torch.einsum('pijq, pxyz, qxyz -> ijxyz', C0, n, n)
    A_e = torch.einsum('piq, pxyz, qxyz -> ixyz', e0, n, n)
    A_d = torch.einsum('pq, pxyz, qxyz -> xyz', K0, n, n)

    # make the first element one
    A_d[0, 0, 0] = 1.0

    # Total matrix
    A = A_c + 1 / A_d * torch.einsum('ixyz, jxyz -> ijxyz', A_e, A_e)

    # Find the inverse of A
    G = torch.einsum('ijxyz -> jixyz', A)

    adjG = torch.empty_like(A, dtype=torch.float32)
    adjG[0,0] = G[1,1] * G[2,2] - G[1,2] * G[2,1]
    adjG[0,1] = -(G[1,0] * G[2,2] - G[1,2] * G[2,0])
    adjG[0,2] = G[1,0] * G[2,1] - G[1,1] * G[2,0]
    adjG[1,0] = -(G[0,1] * G[2,2] - G[0,2] * G[2,1])
    adjG[1,1] = G[0,0] * G[2,2] - G[0,2] * G[2,0]
    adjG[1,2] = -(G[0,0] * G[2,1] - G[0,1] * G[2,0])
    adjG[2,0] = G[0,1] * G[1,2] - G[0,2] * G[1,1]
    adjG[2,1] = -(G[0,0] * G[1,2] - G[0,2] * G[1,0])
    adjG[2,2] = G[0,0] * G[1,1] - G[0,1] * G[1,0]

    # Determinant of A
    detG = A[0,0] * (A[1,1]*A[2,2]-A[1,2]*A[2,1]) \
              - A[0,1] * (A[1,0]*A[2,2]-A[1,2]*A[2,0]) \
                  + A[0,2] * (A[1,0]*A[2,1]-A[1,1]*A[2,0])
    detG[0,0,0] = 1.0

    # Inverse of A
    invA = adjG/detG

    # Green's function of the stiffness tensor in Fourier space
    G_elas = torch.zeros((3, 3, 3, 3, Nx, Ny, Nz), dtype=torch.float32).to(device)
    for i,j,k,l in itertools.product(range(3), repeat=4):
        G_elas[i,j,k,l] = 0.25*(invA[i,l]*n[k]*n[j]+invA[j,l]*n[k]*n[i]+invA[i,k]*n[l]*n[j]+invA[j,k]*n[i]*n[l])

    # Green's function of the piezoelectric tensor in Fourier space
    G_piezo = torch.zeros((3, 3, 3, Nx, Ny, Nz), dtype=torch.float32).to(device)
    for k,i,j in itertools.product(range(3), repeat=3):
        G_piezo[k,i,j] = 1/(2*A_d)*n[k]*(invA[i,0]*n[j] + invA[j,0]*n[i])*A_e[0] + 1/(2*A_d)*n[k]*(invA[i,1]*n[j] + invA[j,1]*n[i])*A_e[1] + 1/(2*A_d)*n[k]*(invA[i,2]*n[j] + invA[j,2]*n[i])*A_e[2]

    # Green's function of the dielectric tensor in Fourier space
    aga = torch.einsum('i...,ij...,j...',A_e,invA,A_e)
    G_dielec = 1/(A_d**2)*(aga-A_d)*torch.einsum('ixyz,jxyz->ijxyz',n,n)

    # delete unused tensors - save more memory
    del aga, invA, G, adjG, A_c, A_e, A_d

    # return the greens tensors
    return G_elas, G_piezo, G_dielec
