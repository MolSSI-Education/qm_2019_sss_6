import numpy as np
from NobleGasModel import NobleGasModel
from scf import scf
from mp2 import MP2
atomic_coordinates = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 5.0]])
# Derived from user input
number_of_atoms = len(atomic_coordinates)

 # Argon parameters - these would change for other noble gases.
model_parameters = {
    'r_hop' : 3.1810226927827516,
    't_ss' : 0.03365982238611262,
    't_sp' : -0.029154833035109226,
    't_pp1' : -0.0804163845390335,
    't_pp2' : -0.01393611496959445,
    'r_pseudo' : 2.60342991362958,
    'v_pseudo' : 0.022972992186364977,
    'dipole' : 2.781629275106456,
    'energy_s' : 3.1659446174413004,
    'energy_p' : -2.3926873325346554,
    'coulomb_s' : 0.3603533286088998,
    'coulomb_p' : -0.003267991835806299
    }
ionic_charge = 6
orbital_types = ['s', 'px', 'py', 'pz']
orbitals_per_atom = len(orbital_types)
p_orbitals = orbital_types[1:]
vec = {'px': [1, 0, 0], 'py': [0, 1, 0], 'pz': [0, 0, 1]}
orbital_occupation = { 's':0, 'px':1, 'py':1, 'pz':1 }

my_model = NobleGasModel(atomic_coordinates,model_parameters,ionic_charge,orbital_types,orbitals_per_atom,vec,orbital_occupation)
interaction_matrix,chi_tensor,hamiltonian_matrix,density_matrix,energy_ion = my_model.kernel()
my_scf = scf(hamiltonian_matrix,interaction_matrix,density_matrix,chi_tensor,energy_ion,ionic_charge,orbitals_per_atom)
print("SCF Energy: " + str(my_scf.kernel()))
my_mp2= MP2(my_scf)
print("MP2 Energy" + str(my_mp2.kernel()))


