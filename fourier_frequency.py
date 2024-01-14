import torch

def fourier_space_torch(Nx:int, dx:float, Ny, dy, Nz, dz, device):

    kx = (2.0 * torch.tensor([torch.pi]) * torch.fft.fftfreq(Nx, dx)).to(device)

    ky = (2.0 * torch.tensor([torch.pi]) * torch.fft.fftfreq(Ny, dy)).to(device)

    kz = (2.0 * torch.tensor([torch.pi]) * torch.fft.fftfreq(Nz, dz)).to(device)

    kx_grid, ky_grid, kz_grid = torch.meshgrid(kx, ky, kz, indexing = 'ij')

    freq = torch.stack((kx_grid, ky_grid, kz_grid))

    return freq
