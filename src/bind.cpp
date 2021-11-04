#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/stl_bind.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

#include "henon.h"

namespace py = pybind11;

PYBIND11_MODULE(henon_map_engine, m)
{
    py::class_<gpu_henon>(m, "gpu_henon")
        .def(py::init<
                 const std::vector<double> &,
                 const std::vector<double> &,
                 const std::vector<double> &,
                 const std::vector<double> &,
                 const double &,
                 const double &>(),
             py::arg("x_init"), py::arg("px_init"), py::arg("y_init"),
             py::arg("py_init"), py::arg("omega_x"), py::arg("omega_y"))

        .def("reset", &gpu_henon::reset, "Resets the engine")
        .def("track", &gpu_henon::track, "Tracks the particles",
             py::arg("n_turns"), py::arg("epsilon"), py::arg("mu"),
             py::arg("barrier") = 100.0, py::arg("kick_module") = NAN,
             py::arg("kick_sigma") = NAN, py::arg("inverse") = false,
             py::arg("modulation_kind") = "sps", py::arg("omega_0") = NAN)

        .def("get_x", &gpu_henon::get_x, "Returns the x coordinates",
             py::return_value_policy::copy)
        .def("get_y", &gpu_henon::get_y, "Returns the y coordinates",
             py::return_value_policy::copy)
        .def("get_px", &gpu_henon::get_px, "Returns the x momenta",
             py::return_value_policy::copy)
        .def("get_py", &gpu_henon::get_py, "Returns the y momenta",
             py::return_value_policy::copy)

        .def("get_steps", &gpu_henon::get_steps,
             "Returns the number of steps before particle loss", py::return_value_policy::copy)

        .def("get_omega_x", &gpu_henon::get_omega_x,
             "Returns the x working tune", py::return_value_policy::copy)
        .def("get_omega_y", &gpu_henon::get_omega_y,
             "Returns the y working tune", py::return_value_policy::copy)
        .def("get_global_steps", &gpu_henon::get_global_steps,
             "Returns the number of global steps performed by the engine",
             py::return_value_policy::copy)

        .def("set_omega_x", &gpu_henon::set_omega_x,
             "Sets the x working tune", py::arg("omega_x"))
        .def("set_omega_y", &gpu_henon::set_omega_y,
             "Sets the y working tune", py::arg("omega_y"))

        .def("set_x", &gpu_henon::set_x,
             "Sets the x coordinates", py::arg("x"))
        .def("set_y", &gpu_henon::set_y,
             "Sets the y coordinates", py::arg("y"))
        .def("set_px", &gpu_henon::set_px,
             "Sets the x momenta", py::arg("px"))
        .def("set_py", &gpu_henon::set_py,
             "Sets the y momenta", py::arg("py"))

        .def("set_steps",
             static_cast<void (gpu_henon::*)(unsigned int)>(&gpu_henon::set_steps),
             "Sets the number of steps before particle loss", py::arg("steps"))
        .def("set_steps",
             static_cast<void (gpu_henon::*)(std::vector<unsigned int>)>(&gpu_henon::set_steps),
             "Sets the number of steps before particle loss", py::arg("steps"))
        .def("set_global_steps", &gpu_henon::set_global_steps,
             "Sets the number of global steps performed by the engine",
             py::arg("global_steps"));

    py::class_<cpu_henon>(m, "cpu_henon")
        .def(py::init<
                 const std::vector<double> &,
                 const std::vector<double> &,
                 const std::vector<double> &,
                 const std::vector<double> &,
                 const double &,
                 const double &>(),
             py::arg("x_init"), py::arg("px_init"), py::arg("y_init"),
             py::arg("py_init"), py::arg("omega_x"), py::arg("omega_y"))

        .def("reset", &cpu_henon::reset, "Resets the engine")
        .def("track", &cpu_henon::track, "Tracks the particles",
             py::arg("n_turns"), py::arg("epsilon"), py::arg("mu"),
             py::arg("barrier") = 100.0, py::arg("kick_module") = NAN,
             py::arg("kick_sigma") = NAN, py::arg("inverse") = false,
             py::arg("modulation_kind") = "sps", py::arg("omega_0") = NAN)

        .def("get_x", &cpu_henon::get_x, "Returns the x coordinates",
             py::return_value_policy::copy)
        .def("get_y", &cpu_henon::get_y, "Returns the y coordinates",
             py::return_value_policy::copy)
        .def("get_px", &cpu_henon::get_px, "Returns the x momenta",
             py::return_value_policy::copy)
        .def("get_py", &cpu_henon::get_py, "Returns the y momenta",
             py::return_value_policy::copy)

        .def("get_steps", &cpu_henon::get_steps,
             "Returns the number of steps before particle loss", py::return_value_policy::copy)

        .def("get_omega_x", &cpu_henon::get_omega_x,
             "Returns the x working tune", py::return_value_policy::copy)
        .def("get_omega_y", &cpu_henon::get_omega_y,
             "Returns the y working tune", py::return_value_policy::copy)
        .def("get_global_steps", &cpu_henon::get_global_steps,
             "Returns the number of global steps performed by the engine",
             py::return_value_policy::copy)

        .def("set_omega_x", &cpu_henon::set_omega_x,
             "Sets the x working tune", py::arg("omega_x"))
        .def("set_omega_y", &cpu_henon::set_omega_y,
             "Sets the y working tune", py::arg("omega_y"))

        .def("set_x", &cpu_henon::set_x,
             "Sets the x coordinates", py::arg("x"))
        .def("set_y", &cpu_henon::set_y,
             "Sets the y coordinates", py::arg("y"))
        .def("set_px", &cpu_henon::set_px,
             "Sets the x momenta", py::arg("px"))
        .def("set_py", &cpu_henon::set_py,
             "Sets the y momenta", py::arg("py"))

        .def("set_steps",
             static_cast<void (cpu_henon::*)(unsigned int)>(&cpu_henon::set_steps),
             "Sets the number of steps before particle loss", py::arg("steps"))
        .def("set_steps",
             static_cast<void (cpu_henon::*)(std::vector<unsigned int>)>(&cpu_henon::set_steps),
             "Sets the number of steps before particle loss", py::arg("steps"))
        .def("set_global_steps", &cpu_henon::set_global_steps,
             "Sets the number of global steps performed by the engine",
             py::arg("global_steps"));
}
