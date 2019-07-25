import numpy as np

class NobleGasModel():

    def __init__(self,coords,params,ionic_charge,orbital_types,orbitals_per_atom,vec,orbital_occupation):
        self.atomic_coordinates = coords
        self.model_parameters = params
        self.ionic_charge = ionic_charge
        self.orbital_types = orbital_types
        self.p_orbitals=orbital_types[1:]
        self.orbitals_per_atom = orbitals_per_atom
        self.vec= vec
        self.orbital_occupation = orbital_occupation
        self.number_of_atoms = len(self.atomic_coordinates)
 
    def atom(self,ao_index):
        '''Returns the atom index part of an atomic orbital index.'''
        return ao_index // self.orbitals_per_atom

    def orb(self,ao_index):
        '''Returns the orbital type of an atomic orbital index.'''
        orb_index = ao_index % self.orbitals_per_atom
        return self.orbital_types[orb_index]

    def ao_index(self,atom_p, orb_p):
        '''Returns the atomic orbital index for a given atom index and orbital type.
        
        Parameters
        ----------
        atom_p: int
            Index of atom
        orb_p: str
            orbital type

        Returns
        -------
        p: int
            atomic orbital
        '''
        p = atom_p * self.orbitals_per_atom
        p += self.orbital_types.index(orb_p)
        return p

    def calculate_hamiltonian_matrix(self,atomic_coordinates, model_parameters):
        '''Returns the 1-body Hamiltonian matrix for an input list of atomic coordinates.

        Parameters
        -----------
        atomic_coordinates : np.array
            an array containing coordinates of atoms
        model_parameters : dict
            a dictionary that contains parameters in this semi-empirical model

        Return
        -----------
        hamiltonian_matrix : np.array

        '''
        ndof = len(atomic_coordinates) * self.orbitals_per_atom
        hamiltonian_matrix = np.zeros((ndof, ndof))
        potential_vector = self.calculate_potential_vector(atomic_coordinates,
                                                    model_parameters)
        for p in range(ndof):
            for q in range(ndof):
                if self.atom(p) != self.atom(q):
                    r_pq = atomic_coordinates[self.atom(p)] - atomic_coordinates[self.atom(
                        q)]
                    hamiltonian_matrix[p, q] = self.hopping_energy(
                        self.orb(p), self.orb(q), r_pq, model_parameters)
                if self.atom(p) == self.atom(q):
                    if p == q and self.orb(p) == 's':
                        hamiltonian_matrix[p, q] += model_parameters['energy_s']
                    if p == q and self.orb(p) in self.p_orbitals:
                        hamiltonian_matrix[p, q] += model_parameters['energy_p']
                    for orb_r in self.orbital_types:
                        r = self.ao_index(self.atom(p), orb_r)
                        hamiltonian_matrix[p, q] += (
                            self.chi_on_atom(self.orb(p), self.orb(q), orb_r, model_parameters) *
                            potential_vector[r])
        return hamiltonian_matrix

    def calculate_atomic_density_matrix(self,atomic_coordinates):
        '''Returns a trial 1-electron density matrix for an input list of atomic coordinates.
        Parameters
       ------------
       atomic_coordinates : np.array
        an array containing coordinates of atoms
       
       Return
       ------------
       density_matrix : np.array
        '''
        ndof = len(atomic_coordinates) * self.orbitals_per_atom
        density_matrix = np.zeros((ndof, ndof))
        for p in range(ndof):
            density_matrix[p, p] = self.orbital_occupation[orb(p)]
        return density_matrix

    def calculate_chi_tensor(self,atomic_coordinates, model_parameters):
        '''Returns the chi tensor for an input list of atomic coordinates

        Parameters
        -----------
        atomic_coordinates : np.array, float
        model_parameters : dictionary
        ndof : Number of degrees of freedom of the system, positive integer
        chi_tensor : multidimensional array (related to vectors)
        orb_q : key
        orb_r : key
        r = 
        orbital_types: dictionary
    '''

        ndof = len(atomic_coordinates) * self.orbitals_per_atom
        chi_tensor = np.zeros((ndof, ndof, ndof))
        for p in range(ndof):
            for orb_q in self.orbital_types:
                q = self.ao_index(self.atom(p), orb_q)
                for orb_r in self.orbital_types:
                    r = self.ao_index(self.atom(p), orb_r)
                    chi_tensor[p, q, r] = self.chi_on_atom(self.orb(p), self.orb(q), self.orb(r),
                                                    model_parameters)
        return chi_tensor

    def calculate_interaction_matrix(self,atomic_coordinates, model_parameters):
        '''Returns the electron-electron interaction energy matrix for an input list of atomic coordinates.

        Inputs
        ------
        atomic_coordinates : np.array
            an array that contains the coordinates of atoms
        model_parameters : dict
            a dictionary that contains parameters in this semi-empirical model

        Outputs
        -------
        interaction_matrix : np.array
            an array that whose elements are electron-electron interaction energies

        '''
        ndof = len(atomic_coordinates)*self.orbitals_per_atom
        interaction_matrix = np.zeros( (ndof,ndof) )
        for p in range(ndof):
            for q in range(ndof):
                if self.atom(p) != self.atom(q):
                    r_pq = atomic_coordinates[self.atom(p)] - atomic_coordinates[self.atom(q)]
                    interaction_matrix[p,q] = self.coulomb_energy(self.orb(p), self.orb(q), r_pq)
                if p == q and self.orb(p) == 's':
                    interaction_matrix[p,q] = model_parameters['coulomb_s']
                if p == q and self.orb(p) in self.p_orbitals:
                    interaction_matrix[p,q] = model_parameters['coulomb_p']
        return interaction_matrix

    def calculate_energy_ion(self,atomic_coordinates):
        energy_ion = 0.0
        for i, r_i in enumerate(atomic_coordinates):
            for j, r_j in enumerate(atomic_coordinates):
                if i < j:
                    energy_ion += (self.ionic_charge**2) * self.coulomb_energy(
                        's', 's', r_i - r_j)
        return energy_ion

    def coulomb_energy(self,o1, o2, r12):
        '''Returns the Coulomb matrix element for a pair of multipoles of type o1 & o2 separated by a vector r12.
        
        Parameters
        ----------
        o1: str
            First multipole type
        o2: str
            Second multipole type
        r12: np.array
            Difference vector between multipole coordinates
        
        Returns
        -------
        ans: float
            Calculated coulomb matrix element for pair of multipoles
        '''

        r12_length = np.linalg.norm(r12)
        if o1 == 's' and o2 == 's':
            ans = 1.0 / r12_length
        if o1 == 's' and o2 in self.p_orbitals:
            ans = np.dot(self.vec[o2], r12) / r12_length**3
        if o2 == 's' and o1 in self.p_orbitals:
            ans = -1 * np.dot(self.vec[o1], r12) / r12_length**3
        if o1 in self.p_orbitals and o2 in self.p_orbitals:
            ans = (
                np.dot(self.vec[o1], self.vec[o2]) / r12_length**3 -
                3.0 * np.dot(self.vec[o1], r12) * np.dot(self.vec[o2], r12) / r12_length**5)
        return ans
    
    def calculate_potential_vector(self,atomic_coordinates, model_parameters):
        ndof = len(atomic_coordinates) * self.orbitals_per_atom
        potential_vector = np.zeros(ndof)
        for p in range(ndof):
            potential_vector[p] = 0.0
            for atom_i, r_i in enumerate(atomic_coordinates):
                r_pi = atomic_coordinates[self.atom(p)] - r_i
                if atom_i != self.atom(p):
                    potential_vector[p] += (
                        self.pseudopotential_energy(self.orb(p), r_pi, model_parameters) -
                        self.ionic_charge * self.coulomb_energy(self.orb(p), 's', r_pi))
        return potential_vector

    def chi_on_atom(self,o1, o2, o3, model_parameters):
        '''Returns the value of the chi tensor for 3 orbital indices on the same atom.

        Inputs
        ------
        o1 : string
            orbital type
        o2 : string
            orbital type
        o3 : string
            orbital type
        model_parameters : dict
            a dictionary that contains parameters in this semi-empirical model

        Outputs
        -------
        value of chi_tensor : float
        '''
        if o1 == o2 and o3 == 's':
            return 1.0
        if o1 == o3 and o3 in self.p_orbitals and o2 == 's':
            return model_parameters['dipole']
        if o2 == o3 and o3 in self.p_orbitals and o1 == 's':
            return model_parameters['dipole']
        return 0.0

    def pseudopotential_energy(self,o, r, model_parameters):
        '''Returns the energy of a pseudopotential between a multipole of type o and an atom separated by a vector r.

        Parameters
        ----------
        o : str
            a string of the type of the orbital
        r : np.array
            difference vector between atoms

        Return
        -----------
        pseudo_vector = np.array
            calcualte the pseudo energy

        '''
        ans = model_parameters['v_pseudo']
        r_rescaled = r / model_parameters['r_pseudo']
        ans *= np.exp(1.0 - np.dot(r_rescaled, r_rescaled))
        if o in self.p_orbitals:
            ans *= -2.0 * np.dot(self.vec[o], r_rescaled)
        return ans
    
    def hopping_energy(self,o1, o2, r12, model_parameters):
        r12_rescaled = r12 / model_parameters['r_hop']
        r12_length = np.linalg.norm(r12_rescaled)
        ans = np.exp( 1.0 - r12_length**2 )
        if o1 == 's' and o2 == 's':
            ans *= model_parameters['t_ss']
        if o1 == 's' and o2 in self.p_orbitals:
            ans *= np.dot(self.vec[o2], r12_rescaled) * model_parameters['t_sp']
        if o2 == 's' and o1 in self.p_orbitals:
            ans *= -1.0*np.dot(self.vec[o1], r12_rescaled)* model_parameters['t_sp']
        if o1 in self.p_orbitals and o2 in self.p_orbitals:
            ans *= ( (r12_length**2) * np.dot(self.vec[o1], self.vec[o2]) * model_parameters['t_pp2']
                    - np.dot(self.vec[o1], r12_rescaled) * np.dot(self.vec[o2], r12_rescaled)
                    * ( model_parameters['t_pp1'] + model_parameters['t_pp2'] ) )
        return ans

    def kernel(self):
        self.interaction_matrix = self.calculate_interaction_matrix(self.atomic_coordinates, self.model_parameters)
        self.chi_tensor=self.calculate_chi_tensor(self.atomic_coordinates, self.model_parameters)
        self.hamiltonian_matrix = self.calculate_hamiltonian_matrix(self.atomic_coordinates, self.model_parameters)
        self.density_matrix = self.calculate_atomic_density_matrix(self.atomic_coordinates)
        self.energy_ion = self.calculate_energy_ion(self.atomic_coordinates)
        return self.interaction_matrix,self.chi_tensor,self.hamiltonian_matrix,self.density_matrix,self.energy_ion


if __name__ == "__main__":
    ## --------------------
    ## Noble Gas Parameters
    ## --------------------
    ionic_charge = 6
    orbital_types = ['s', 'px', 'py', 'pz']
    orbitals_per_atom = len(orbital_types)
    p_orbitals = orbital_types[1:]
    vec = {'px': [1, 0, 0], 'py': [0, 1, 0], 'pz': [0, 0, 1]}
    orbital_occupation = { 's':0, 'px':1, 'py':1, 'pz':1 }    
    # User input
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