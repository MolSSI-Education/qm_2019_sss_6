#include <iostream>
#include <vector>
#include <string>
#include <list>
// #include <Eigen/Dense>
#include <map>

const int orbitals_per_atom = 4;

double model_parameters = 1.0;


bool substring_in_string(std::string str, std::string substr);

int atom(int int_ao_index, int orbitals_per_atom);

std::string orb(int int_ao_index, int orbitals_per_atom);

int ao_index(int atom_p, std::string orb_p, int orbitals_per_atom);

double chi_on_atom(std::string o1, std::string o2, std::string o3, std::map<std::string, float> model_parameters);

Eigen::MatrixXd calculate_fock_matrix_fast(Eigen::MatrixXd hamiltonian_matrix,
  Eigen::MatrixXd interaction_matrix, Eigen::MatrixXd density_matrix,
  std::map<std::string, float> model_parameters, std::vector<std::string> orbital_types);

  
