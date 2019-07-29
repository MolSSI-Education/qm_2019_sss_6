#include <iostream>
#include <vector>
#include <string>
#include <list>
#include <Eigen/Dense>
#include <map>
#include "scf.hpp"


// template <class T>
// bool collection_contains(std::vector<T> my_collection, T my_element)
// {
//   bool found = (std::find(my_collection.begin(), my_collection.end(), my_element) != my_collection.end());
//   return found;
// }

bool substring_in_string(std::string str, std::string substr)
{
  if(str.std::string::find(substr) != std::string::npos)
  {
    return true;
  }
  else
  {
    return false;
  }
}

int atom(int int_ao_index, int orbitals_per_atom)
{
  return int_ao_index/orbitals_per_atom;
}

std::string orb(int int_ao_index, int orbitals_per_atom)
{
  int int_orb =  int_ao_index%orbitals_per_atom;
  if(int_orb == 0)
  {
    return "s";
  }

  else if(int_orb == 1)
  {
    return "px";
  }

  else if(int_orb == 2)
  {
    return "py";
  }

  else
  {
    return "pz";
  }
}

int ao_index(int atom_p, std::string orb_p, int orbitals_per_atom)
{
  int p = atom_p*orbitals_per_atom;
  int p_type_int;
  if(orb_p == "s")
  {
    p_type_int = 0;
  }

  else if(orb_p == "px")
  {
    p_type_int = 1;
  }

  else if(orb_p == "py")
  {
    p_type_int = 2;
  }

  else
  {
    p_type_int = 3;
  }
  p = atom_p*orbitals_per_atom;
  p += p_type_int;
  return p;
}

double chi_on_atom(std::string o1, std::string o2, std::string o3, std::map<std::string, float> model_parameters)
{
  if(o1 == o2 && o3 == "s")
  {
    return 1.0;
  }

  if(o1 == o3 && substring_in_string(o3, "p") && o2 == "s")
  {
    return model_parameters["dipole"];
  }

  if(o2 == o3 && substring_in_string(o3, "p") && o1 == "s")
  {
    return model_parameters["dipole"];
  }
  return 0.0;
}




Eigen::MatrixXd calculate_fock_matrix_fast(Eigen::MatrixXd hamiltonian_matrix,
  Eigen::MatrixXd interaction_matrix, Eigen::MatrixXd density_matrix,
  std::map<std::string, float> model_parameters, std::vector<std::string> orbital_types)
{
    size_t ndof = hamiltonian_matrix.size();
    Eigen::MatrixXd fock_matrix = hamiltonian_matrix;
    // # Hartree potential term
    for(int i_orb_p = 0; i_orb_p < ndof; ++i_orb_p)
    {
      std::string orb_p = orb(i_orb_p, orbitals_per_atom);

      for(int i_orb_q = 0; i_orb_q < orbital_types.size(); ++i_orb_q)
      {
        std::string orb_q = orb(i_orb_q, orbitals_per_atom);

        for(int i_orb_t = 0; i_orb_t < orbital_types.size(); ++i_orb_t)
        {
          std::string orb_t = orb(i_orb_t, orbitals_per_atom);

          double chi_pqt = chi_on_atom(orb_p, orb_q, orb_t, model_parameters);

          for(int i_orb_r = 0; i_orb_r < ndof; ++i_orb_r) //for r in range(ndof):
          {
            std::string orb_r = orb(i_orb_r, orbitals_per_atom);

            for(int i_orb_s = 0; i_orb_s < orbital_types.size(); ++i_orb_s) //for orb_s in orbital_types:
            {
              std::string orb_s = orb(i_orb_s, orbitals_per_atom);

              for(int i_orb_u = 0; i_orb_u < orbital_types.size(); ++i_orb_u) //for orb_u in orbital_types:
              {
                std::string orb_u = orb(i_orb_u, orbitals_per_atom);

                double chi_rsu = chi_on_atom(orb_r, orb_s, orb_u, model_parameters);

                fock_matrix(i_orb_p, i_orb_q) += 2.0* chi_pqt * chi_rsu * interaction_matrix(i_orb_t, i_orb_u) * density_matrix(i_orb_r, i_orb_s);
              }
            }
          }
        }
      }
    }


    for(int i_orb_p = 0; i_orb_p < ndof; ++i_orb_p)
    {
      std::string orb_p = orb(i_orb_p, orbitals_per_atom);

      for(int i_orb_s = 0; i_orb_s < orbital_types.size(); ++i_orb_s)
      {
        std::string orb_s = orb(i_orb_s, orbitals_per_atom);

        for(int i_orb_u = 0; i_orb_u < orbital_types.size(); ++i_orb_u)
        {
          std::string orb_u = orb(i_orb_u, orbitals_per_atom);

          double chi_psu = chi_on_atom(orb_p, orb_s, orb_u, model_parameters);

          for(int i_orb_q = 0; i_orb_q < ndof; ++i_orb_q)
          {
            std::string orb_q = orb(i_orb_q, orbitals_per_atom);

            for(int i_orb_r = 0; i_orb_r < orbital_types.size(); ++i_orb_r)
            {
              std::string orb_r = orb(i_orb_r, orbitals_per_atom);

              for(int i_orb_t = 0; i_orb_t < orbital_types.size(); ++i_orb_t)
              {
                std::string orb_t = orb(i_orb_t, orbitals_per_atom);

                double chi_rqt = chi_on_atom(orb_r, orb_q, orb_t, model_parameters);
                fock_matrix(i_orb_p, i_orb_q) -= chi_rqt * chi_psu * interaction_matrix(i_orb_t, i_orb_u) * density_matrix(i_orb_r, i_orb_s);
              }
            }
          }
        }
      }
    }

    return fock_matrix;
}






int main()
{
  std::cout << orb(1, 4) << std::endl;
  // std::vector<std::string> orbital_types;
  // orbital_types.push_back("s");
  // orbital_types.push_back("px");
  // orbital_types.push_back("py");
  // orbital_types.push_back("pz");
  //
  // std::vector<std::string> p_orbitals;
  // p_orbitals.push_back("px");
  // p_orbitals.push_back("py");
  // p_orbitals.push_back("pz");
  return 0;
}
