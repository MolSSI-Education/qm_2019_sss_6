#include <Eigen/Dense>
#include <algorithm>
#include <vector>

int atom(int ao_index);
int indexOf(std::vector<std::string> my_list, std::string element);
bool list_contains(std::vector<std::string>  my_list, std::string element);
std::string orb(int ao_index, std::vector<std::string> orbital_types);
int ao_index(int atom_p, std::string orb_p, std::vector<std::string>  orbital_types);
float chi_on_atom(std::string o1, std::string o2, std::string o3, std::vector<std::string> p_orbitals,double dipole);
Eigen::MatrixXd fast_fock_matrix(Eigen::MatrixXd hamiltonian, Eigen::MatrixXd interaction, Eigen::MatrixXd rho,double dipole);
int main(void);
