import torch

def fourier_frequencies(Nx:int, dx:float, Ny:int, dy:float, Nz:int, dz:float, device:torch.device):
    """
    Computes the FFT sample frequencies for a 3D

    Nx, Ny, Nz - number of grid points in each direction

    dx, dy, dz - grid spacing in each direction
    
    device - torch device; "cpu" or "cuda"
    """
    # in x-direction
    kx = (2.0 * torch.tensor([torch.pi]) * torch.fft.fftfreq(Nx, dx)).to(device)

    # in y-direction
    ky = (2.0 * torch.tensor([torch.pi]) * torch.fft.fftfreq(Ny, dy)).to(device)

    # in z-direction
    kz = (2.0 * torch.tensor([torch.pi]) * torch.fft.fftfreq(Nz, dz)).to(device)

    # Create grids of coordinates
    kx_grid, ky_grid, kz_grid = torch.meshgrid(kx, ky, kz, indexing = 'ij')

    # Concatenates a sequence of tensors
    freq = torch.stack((kx_grid, ky_grid, kz_grid))

    return freq
