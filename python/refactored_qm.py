class scf():
    def __init__(self,hamiltonian_matrix,interaction_matrix,density_matrix,chi_tensor,atomic_coordinates):
        self.hamiltonian_matrix = hamiltonian_matrix
        self.interaction_matrix = interaction_matrix
        self.density_matrix = density_matrix
        self.chi_tensor = chi_tensor
        self.atomic_coordinates = atomic_coordinates
        self.converged = False
    def scf_cycle(self,hamiltonian_matrix, interaction_matrix, density_matrix,
                chi_tensor, max_scf_iterations = 100,
                mixing_fraction = 0.25, convergence_tolerance = 1e-4):
        """Returns converged density & Fock matrices defined by the input Hamiltonian, interaction, & density matrices.

        Parameters
        ----------
        Initial Hamiltonian : np.array
            Defines the initial orbital energy and phase space.
        Interaction Matrix: ndarray
            Defines the initial interaction between different atoms.
        Density Matrix: ndarray
            Defines the electron density on atoms.

        Returns
        -------
        If SCF converges, then returns modified density matrix and modified fock matrix.
    """
        old_density_matrix = density_matrix.copy()
        for iteration in range(max_scf_iterations):
            new_fock_matrix = calculate_fock_matrix(hamiltonian_matrix, interaction_matrix, old_density_matrix, chi_tensor)
            new_density_matrix = calculate_density_matrix(new_fock_matrix)

            error_norm = np.linalg.norm( old_density_matrix - new_density_matrix )
            if error_norm < convergence_tolerance:
                return new_density_matrix, new_fock_matrix

            old_density_matrix = (mixing_fraction * new_density_matrix
                                + (1.0 - mixing_fraction) * old_density_matrix)
        print("WARNING: SCF cycle didn't converge")
        return new_density_matrix, new_fock_matrix
    def calculate_energy_ion(self,atomic_coordinates):

        """Returns the ionic contribution to the total energy for an input list of atomic coordinates.


        Parameters
        ----------
        coordinates : np.array
            An array of atomic coordinates.
            Defines initial energy of the ion as 0.0.

        Returns
        -------
        Energy of the ions : int/Float/ndarray
            A float, integer, or ndarray depending on the size of the molecule.
        """

        energy_ion = 0.0
        for i, r_i in enumerate(atomic_coordinates):
            for j, r_j in enumerate(atomic_coordinates):
                if i < j:
                    energy_ion += (ionic_charge**2) * coulomb_energy(
                        's', 's', r_i - r_j)
        return energy_ion
    def calculate_energy_scf(self,hamiltonian_matrix, fock_matrix, density_matrix):
        '''Returns the Hartree-Fock total energy defined by the input Hamiltonian, Fock, & density matrices.

        Inputs
        ------
        hamiltonian_matrix : np.array

        fock_matrix : np.array

        density_matrix : np.array


        Output
        ------
        energy_scf : float
            Hartree-Fock total energy

        '''
        energy_scf = np.einsum('pq,pq', hamiltonian_matrix + fock_matrix,
                            density_matrix)
        return energy_scf
    def calculate_density_matrix(self,fock_matrix):
        '''Returns the 1-electron density matrix defined by the input Fock matrix.

            Parameters
        ------------
        fock_matrix : np.array

        Return
        ------------
        density_matrix : np.array


        '''
        num_occ = (ionic_charge // 2) * np.size(fock_matrix,
                                                0) // orbitals_per_atom
        orbital_energy, orbital_matrix = np.linalg.eigh(fock_matrix)
        occupied_matrix = orbital_matrix[:, :num_occ]
        density_matrix = occupied_matrix @ occupied_matrix.T
        return density_matrix
    def calculate_fock_matrix(self,hamiltonian_matrix, interaction_matrix,
                            density_matrix, chi_tensor):
        '''Returns the Fock matrix defined by the input Hamiltonian, interaction, & density matrices.

        Parameters
        ------------
        hamiltonian_matrix : np.array
        interaction_matrix : np.array
        density_matrix : np.array
        chi_tensor : np.array

        Return
        ------------
        fock_matrix : np.array

        '''
        fock_matrix = hamiltonian_matrix.copy()
        fock_matrix += 2.0 * np.einsum('pqt,rsu,tu,rs',
                                    chi_tensor,
                                    chi_tensor,
                                    interaction_matrix,
                                    density_matrix,
                                    optimize=True)
        fock_matrix -= np.einsum('rqt,psu,tu,rs',
                                chi_tensor,
                                chi_tensor,
                                interaction_matrix,
                                density_matrix,
                                optimize=True)
        return fock_matrix
    def initialize(self):
        self.fock_matrix=self.calculate_fock_matrix(self.hamiltonian_matrix,self.interaction_matrix,self.density_matrix,self.chi_tensor)
        self.density_matrix=self.calculate_density_matrix(self.fock_matrix)
    def kernel(self):
        self.initialize()
        self.density_matrix, self.fock_matrix = self.scf_cycle(self.hamiltonian_matrix, self.interaction_matrix, self.density_matrix,
                self.chi_tensor)
        self.energy_ion = self.calculate_energy_ion(self.atomic_coordinates)
        self.energy_scf = self.calculate_energy_scf(self.hamiltonian_matrix, self.fock_matrix, self.density_matrix)
        self.total_energy=energy_ion+energy_scf
        return self.total_energy


class MP2():
    def __init__(self,scf):
        self.scf=scf

class Semi_Empirical_Model():
    def atom(ao_index):
        '''Returns the atom index part of an atomic orbital index.'''
        return ao_index // orbitals_per_atom
    def orb(ao_index):
        '''Returns the orbital type of an atomic orbital index.'''
        orb_index = ao_index % orbitals_per_atom
        return orbital_types[orb_index]

    def ao_index(atom_p, orb_p):
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
        p = atom_p * orbitals_per_atom
        p += orbital_types.index(orb_p)
        return p
    def calculate_hamiltonian_matrix(atomic_coordinates, model_parameters):
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
        ndof = len(atomic_coordinates) * orbitals_per_atom
        hamiltonian_matrix = np.zeros((ndof, ndof))
        potential_vector = calculate_potential_vector(atomic_coordinates,
                                                    model_parameters)
        for p in range(ndof):
            for q in range(ndof):
                if atom(p) != atom(q):
                    r_pq = atomic_coordinates[atom(p)] - atomic_coordinates[atom(
                        q)]
                    hamiltonian_matrix[p, q] = hopping_energy(
                        orb(p), orb(q), r_pq, model_parameters)
                if atom(p) == atom(q):
                    if p == q and orb(p) == 's':
                        hamiltonian_matrix[p, q] += model_parameters['energy_s']
                    if p == q and orb(p) in p_orbitals:
                        hamiltonian_matrix[p, q] += model_parameters['energy_p']
                    for orb_r in orbital_types:
                        r = ao_index(atom(p), orb_r)
                        hamiltonian_matrix[p, q] += (
                            chi_on_atom(orb(p), orb(q), orb_r, model_parameters) *
                            potential_vector[r])
        return hamiltonian_matrix
    def calculate_atomic_density_matrix(atomic_coordinates):
    '''Returns a trial 1-electron density matrix for an input list of atomic coordinates.

       Parameters
       ------------
       atomic_coordinates : np.array
        an array containing coordinates of atoms

       Return
       ------------
       density_matrix : np.array

    '''
    ndof = len(atomic_coordinates) * orbitals_per_atom
    density_matrix = np.zeros((ndof, ndof))
    for p in range(ndof):
        density_matrix[p, p] = orbital_occupation[orb(p)]
    return density_matrix

    def calculate_chi_tensor(atomic_coordinates, model_parameters):
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

        ndof = len(atomic_coordinates) * orbitals_per_atom
        chi_tensor = np.zeros((ndof, ndof, ndof))
        for p in range(ndof):
            for orb_q in orbital_types:
                q = ao_index(atom(p), orb_q)
                for orb_r in orbital_types:
                    r = ao_index(atom(p), orb_r)
                    chi_tensor[p, q, r] = chi_on_atom(orb(p), orb(q), orb(r),
                                                    model_parameters)
        return chi_tensor

    def calculate_interaction_matrix(atomic_coordinates, model_parameters):
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
        ndof = len(atomic_coordinates)*orbitals_per_atom
        interaction_matrix = np.zeros( (ndof,ndof) )
        for p in range(ndof):
            for q in range(ndof):
                if atom(p) != atom(q):
                    r_pq = atomic_coordinates[atom(p)] - atomic_coordinates[atom(q)]
                    interaction_matrix[p,q] = coulomb_energy(orb(p), orb(q), r_pq)
                if p == q and orb(p) == 's':
                    interaction_matrix[p,q] = model_parameters['coulomb_s']
                if p == q and orb(p) in p_orbitals:
                    interaction_matrix[p,q] = model_parameters['coulomb_p']
        return interaction_matrix

    def calculate_energy_ion(atomic_coordinates):

    """Returns the ionic contribution to the total energy for an input list of atomic coordinates.


    Parameters
    ----------
    coordinates : np.array
        An array of atomic coordinates.
        Defines initial energy of the ion as 0.0.

    Returns
    -------
    Energy of the ions : int/Float/ndarray
        A float, integer, or ndarray depending on the size of the molecule.
    """

    energy_ion = 0.0
    for i, r_i in enumerate(atomic_coordinates):
        for j, r_j in enumerate(atomic_coordinates):
            if i < j:
                energy_ion += (ionic_charge**2) * coulomb_energy(
                    's', 's', r_i - r_j)
    return energy_ion
