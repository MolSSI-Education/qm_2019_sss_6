#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "fock_fast.hpp"

PYBIND11_MODULE(fock_fast,m)
{
    m.doc() = "This is an example.";
    m.def("fast_fock_matrix",fast_fock_matrix,"Faster fock matrix implemented in C++.");
}

