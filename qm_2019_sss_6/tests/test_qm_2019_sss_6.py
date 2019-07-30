"""
Unit and regression test for the qm_2019_sss_6 package.
"""

# Import package, test suite, and other packages as needed
import qm_2019_sss_6
# import NobleGasModel.py
import pytest
import sys
import numpy as np

atomic_coordinates = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 5.0]])
# Derived from user input
number_of_atoms = len(atomic_coordinates)
orbitals_per_atom = 4
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

my_model = qm_2019_sss_6.NobleGasModel(atomic_coordinates,model_parameters,ionic_charge,orbital_types,orbitals_per_atom,vec,orbital_occupation)

def test_qm_2019_sss_6_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "qm_2019_sss_6" in sys.modules

def test_atom():
    for index in range(orbitals_per_atom):
        assert my_model.atom(index) == 0

def test_orb():
    for index in range(orbitals_per_atom):
        assert my_model.orb(index) == orbital_types[index]

def test_ao_index():
    for index in range(number_of_atoms*orbitals_per_atom):
        assert my_model.ao_index(my_model.atom(index), my_model.orb(index)) == index

def test_calculate_hamiltonian_matrix():
    expected_matrix = np.array([[ 2.3e+00, -1.4e-01, -1.9e-01, -2.4e-01,  6.5e-04,  5.3e-04,  7.1e-04,  8.9e-04],
    [-1.4e-01, -3.2e+00,  0.0e+00,  0.0e+00, -5.3e-04,  2.9e-04,  2.2e-03,  2.7e-03],
    [-1.9e-01,  0.0e+00, -3.2e+00,  0.0e+00, -7.1e-04,  2.2e-03,  1.6e-03,  3.6e-03],
    [-2.4e-01,  0.0e+00,  0.0e+00, -3.2e+00, -8.9e-04,  2.7e-03,  3.6e-03,  3.2e-03],
    [ 6.5e-04, -5.3e-04, -7.1e-04, -8.9e-04,  2.3e+00,  1.4e-01,  1.9e-01,  2.4e-01],
    [ 5.3e-04,  2.9e-04,  2.2e-03,  2.7e-03,  1.4e-01, -3.2e+00,  0.0e+00,  0.0e+00],
    [ 7.1e-04,  2.2e-03,  1.6e-03,  3.6e-03,  1.9e-01,  0.0e+00, -3.2e+00,  0.0e+00],
    [ 8.9e-04,  2.7e-03,  3.6e-03,  3.2e-03,  2.4e-01,  0.0e+00,  0.0e+00, -3.2e+00]])

    assert np.allclose(my_model.calculate_hamiltonian_matrix(atomic_coordinates, model_parameters), expected_matrix, rtol=1e-01)

def test_calculate_atomic_density_matrix():
    expected_matrix = [[0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 0., 0., 0.],
    [0., 0., 1., 0., 0., 0., 0., 0.],
    [0., 0., 0., 1., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 1., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 0.],
    [0., 0., 0., 0., 0., 0., 0., 1.]]

    assert np.allclose(my_model.calculate_atomic_density_matrix(atomic_coordinates), expected_matrix, rtol=1e-01)


def test_calculate_chi_tensor():
    expected_tensor = np.array([[[1.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  2.8, 0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  2.8, 0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  2.8, 0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ]],
#
    [[0.,  2.8, 0.,  0.,  0.,  0.,  0.,  0. ],
    [1.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ]],
#
    [[0.,  0.,  2.8, 0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [1.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ]],
#
    [[0.,  0.,  0.,  2.8, 0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [1.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],],
#
    [[0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  1.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  2.8, 0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  2.8, 0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  2.8]],
#
    [[0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  2.8, 0.,  0. ],
    [0.,  0.,  0.,  0.,  1.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ]],
#
    [[0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  2.8, 0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  1.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ]],
#
    [[0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  2.8],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0. ],
    [0.,  0.,  0.,  0.,  1.,  0.,  0.,  0. ]]])

    assert np.allclose( my_model.calculate_chi_tensor(atomic_coordinates, model_parameters), expected_tensor, rtol=1e-02)



def test_calculate_interaction_matrix():
    expected_matrix = np.array(
    [[ 3.6e-01,  0.0e+00,  0.0e+00,  0.0e+00,  1.4e-01, -8.5e-03, -1.1e-02, -1.4e-02],
    [ 0.0e+00, -3.3e-03,  0.0e+00,  0.0e+00,  8.5e-03,  1.3e-03, -2.0e-03, -2.5e-03],
    [ 0.0e+00,  0.0e+00, -3.3e-03,  0.0e+00,  1.1e-02, -2.0e-03,  1.1e-04, -3.4e-03],
    [ 0.0e+00,  0.0e+00,  0.0e+00, -3.3e-03,  1.4e-02, -2.5e-03, -3.4e-03, -1.4e-03],
    [ 1.4e-01,  8.5e-03,  1.1e-02,  1.4e-02,  3.6e-01,  0.0e+00,  0.0e+00,  0.0e+00],
    [-8.5e-03,  1.3e-03, -2.0e-03, -2.5e-03,  0.0e+00, -3.3e-03,  0.0e+00,  0.0e+00],
    [-1.1e-02, -2.0e-03,  1.1e-04, -3.4e-03,  0.0e+00,  0.0e+00, -3.3e-03,  0.0e+00],
    [-1.4e-02, -2.5e-03, -3.4e-03, -1.4e-03,  0.0e+00,  0.0e+00,  0.0e+00, -3.3e-03]])

    assert np.allclose( my_model.calculate_interaction_matrix(atomic_coordinates, model_parameters), expected_matrix, rtol=1e-01)


# def test_calculate_energy_ion():
#
# def test_coulomb_energy():
#
# def test_calculate_potential_vector():
#
# def test_chi_on_atom():
#
# def test_pseudopotential_energy():
#
# def test_hopping_energy():
#
# def test_kernel():
