

def write_vtk(step, polarization, dx, dy, dz):

    """
    Write a VTK file with polarization vector data.

    Parameters:
        - filename (str): The name of the VTK file to be created.
        - polarization (numpy.ndarray): The polarization vector with shape (3, Nx, Ny, Nz).
        - dx (float): Grid spacing along the x-axis.
        - dy (float): Grid spacing along the y-axis.
        - dz (float): Grid spacing along the z-axis.
    """

     # Ensure the polarization vector has the correct shape
    assert polarization.shape[0] == 3, "Polarization vector should have shape (3, Nx, Ny, Nz)"

    Nx, Ny, Nz = polarization.shape[1], polarization.shape[2], polarization.shape[3]

    with open("time_" + str(step) + ".vtk", 'w') as vtk_file:

        # Write VTK header
        vtk_file.write("# vtk DataFile Version 2.0\n")
        vtk_file.write("Polarization Data\n")
        vtk_file.write("ASCII\n")
        vtk_file.write("DATASET STRUCTURED_GRID\n")
        vtk_file.write(f"DIMENSIONS {Nx} {Ny} {Nz}\n")
        vtk_file.write(f"POINTS {Nx*Ny*Nz} float\n")

        # Write grid point coordinates
        for k in range(Nz):
            for j in range(Ny):
                for i in range(Nx):
                    vtk_file.write(f"{i*dx} {j*dy} {k*dz}\n")

        # Write point data header
        vtk_file.write(f"POINT_DATA {Nx*Ny*Nz}\n")
        vtk_file.write("VECTORS polarization float\n")

        # Write polarization vector data
        for k in range(Nz):
            for j in range(Ny):
                for i in range(Nx):
                    vtk_file.write(f"{polarization[0, i, j, k]} {polarization[1, i, j, k]} {polarization[2, i, j, k]}\n")




def write_to_vtk3D(folder, istep, variable_name, data, nx, ny, nz, dx, dy, dz):

    data_x = data[0]
    data_y = data[1]
    data_z = data[2]

    filename = folder + "/time_"+str(istep)+".vtk"
    file = open(filename, 'w')
    npoin = nx * ny * nz
    file.write('# vtk DataFile Version 3.0\n')
    file.write('vtk output\n')
    file.write('ASCII\n')
    file.write('DATASET STRUCTURED_GRID  \n')
    file.write('DIMENSIONS {} {} {}\n'.format(nx, ny, nz))
    file.write('POINTS {} float\n'.format(npoin))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x = i*dx
                y = j*dy
                z = k*dz
                file.write('{} {} {}\n'.format(x, y, z))
    file.write('POINT_DATA {}\n'.format(npoin))
    file.write('VECTORS\t' + str(variable_name) +  '\tfloat\n')
    # file.write('LOOKUP_TABLE default\n')
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                file.write('{} {} {}\n'.format(data_x[i,j,k], data_y[i,j,k], data_z[i,j,k]))
    file.close()


def write_to_vtk3D_PE(istep, P, E, nx, ny, nz, dx, dy, dz):

    filename = "time_"+str(istep)+".vtk"
    file = open(filename, 'w')
    npoin = nx * ny * nz
    file.write('# vtk DataFile Version 3.0\n')
    file.write('vtk output\n')
    file.write('ASCII\n')
    file.write('DATASET STRUCTURED_GRID\n')
    file.write('DIMENSIONS {} {} {}\n'.format(nx, ny, nz))
    file.write('POINTS {} float\n'.format(npoin))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x = i*dx
                y = j*dy
                z = k*dz
                file.write('{} {} {}\n'.format(x, y, z))
    file.write('POINT_DATA {}\n'.format(npoin))
    variable_name_1 = "Polarization"
    file.write('VECTORS\t' + str(variable_name_1) +  '\tfloat\n')
    # Store polarization field
    # file.write('LOOKUP_TABLE default\n')
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                file.write('{} {} {}\n'.format(P[0][i,j,k], P[1][i,j,k], P[2][i,j,k]))
    # store displacements field
    variable_name_2 = "ElectricField"
    file.write('VECTORS\t' + str(variable_name_2) +  '\tfloat\n')
    # file.write('LOOKUP_TABLE default\n')
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                file.write('{} {} {}\n'.format(E[0][i,j,k], E[1][i,j,k], E[2][i,j,k]))
    file.close()
