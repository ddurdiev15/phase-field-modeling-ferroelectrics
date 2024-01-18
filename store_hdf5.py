"""
All data can be stored as .h5 file

Write_to_HDF5 -> in this function you always need to give 5 KEYWORD ARGUMENTS:

    - folder -> directory to save results
    - step   -> the time step
    - P -> polarization
    - Elas_strain  ->  elastic strain
    - Elec_field  ->  electric field
"""

import h5py

def Write_to_HDF5(folder, step, P, **kwargs):

    print(step)

    hdf = h5py.File(str(folder) + '/results.h5','a')
    time = '/time_'+str(int(step))

    # Polarization tensor
    hdf.create_dataset('Polarization/Px'+str(time), data = P[0])
    hdf.create_dataset('Polarization/Py'+str(time), data = P[1])
    hdf.create_dataset('Polarization/Pz'+str(time), data = P[2])

    for key, value in kwargs.items():

        if key == 'ElasticStrain':

            # Elastic strain tensor
            hdf.create_dataset('Elastic strain/strain_XX'+str(time), data = value[0,0])
            hdf.create_dataset('Elastic strain/strain_XY'+str(time), data = value[0,1])
            hdf.create_dataset('Elastic strain/strain_XZ'+str(time), data = value[0,2])
            hdf.create_dataset('Elastic strain/strain_YY'+str(time), data = value[1,1])
            hdf.create_dataset('Elastic strain/strain_YZ'+str(time), data = value[1,2])
            hdf.create_dataset('Elastic strain/strain_ZZ'+str(time), data = value[2,2])

        elif key == 'ElectricField':

            # Electric field tensor
            hdf.create_dataset('Electric field/Ex'+str(time), data = value[0])
            hdf.create_dataset('Electric field/Ey'+str(time), data = value[1])
            hdf.create_dataset('Electric field/Ez'+str(time), data = value[2])

    hdf.close()
