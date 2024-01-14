
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
