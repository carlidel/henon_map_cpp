#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/stl_bind.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

#include "henon.h"
#include "modulation.h"

bool has_cuda_error_happened() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return true;
    }
    return false;
}


bool is_cuda_device_available() {
    int count;
    cudaGetDeviceCount(&count);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        return false;
    }
    std::cout << "CUDA devices available: " << count << std::endl;
    return count > 0;
}


namespace py = pybind11;

class pyhenon : public henon {
public:
    /* Inherit the constructors */
    using henon::henon;

    /* Trampoline (need one for each virtual function) */
    void reset() override {PYBIND11_OVERRIDE(void, henon, reset, );}
    
    void compute_a_modulation(unsigned int N, double omega_x, double omega_y, std::string modulation_kind = "sps", double omega_0 = NAN, double epsilon = 0.0, unsigned int offset = 0) override
    {
        PYBIND11_OVERRIDE(
            void, 
            henon, 
            compute_a_modulation, 
            N, omega_x, omega_y, modulation_kind, omega_0, epsilon, offset
        );
    }

    void track(
        unsigned int n_turns, double mu, double barrier = 100.0,
        double kick_module = NAN, bool inverse = false) override
    {
        PYBIND11_OVERRIDE_PURE(
            void,
            henon,
            track,
            n_turns, mu, barrier, kick_module, inverse
        );
    }

    std::vector<std::vector<double>> track_MEGNO(
        std::vector<unsigned int> n_turns, double mu,
        double barrier = 100.0,
        double kick_module = NAN,
        bool inverse = false) override
    {
        PYBIND11_OVERRIDE_PURE(
            std::vector<std::vector<double>>,
            henon,
            track_MEGNO,
            n_turns, mu, barrier, kick_module, inverse
        );
    }

    std::vector<double> get_x() const override { PYBIND11_OVERRIDE(std::vector<double>, henon, get_x); }
    std::vector<double> get_px() const override { PYBIND11_OVERRIDE(std::vector<double>, henon, get_px); }
    std::vector<double> get_y() const override { PYBIND11_OVERRIDE(std::vector<double>, henon, get_y); }
    std::vector<double> get_py() const override { PYBIND11_OVERRIDE(std::vector<double>, henon, get_py); }
    std::vector<unsigned int> get_steps() const override { PYBIND11_OVERRIDE(std::vector<unsigned int>, henon, get_steps); }

    void set_x(std::vector<double> x) override{PYBIND11_OVERRIDE(void, henon, set_x, x);}
    void set_px(std::vector<double> px) override {PYBIND11_OVERRIDE(void, henon, set_px, px);}
    void set_y(std::vector<double> y) override {PYBIND11_OVERRIDE(void, henon, set_y, y);}
    void set_py(std::vector<double> py) override {PYBIND11_OVERRIDE(void, henon, set_py, py);}
    void set_steps(std::vector<unsigned int> steps) override {PYBIND11_OVERRIDE(void, henon, set_steps, steps);}
    void set_steps(unsigned int unique_step) override {PYBIND11_OVERRIDE(void, henon, set_steps, unique_step);}
};

class pycpu_henon : public cpu_henon {
public:
    /* Inherit the constructors */
    using cpu_henon::cpu_henon;

    /* Trampoline (need one for each virtual function) */
    void reset() override {PYBIND11_OVERRIDE(void, cpu_henon, reset, );}
    
    void compute_a_modulation(unsigned int N, double omega_x, double omega_y, std::string modulation_kind = "sps", double omega_0 = NAN, double epsilon = 0.0, unsigned int offset = 0) override
    {
        PYBIND11_OVERRIDE(
            void, 
            cpu_henon, 
            compute_a_modulation, 
            N, omega_x, omega_y, modulation_kind, omega_0, epsilon, offset
        );
    }

    void track(unsigned int n_turns, double mu, double barrier = 100.0, double kick_module = NAN, bool inverse = false) override
    {
        PYBIND11_OVERRIDE(
            void,
            cpu_henon,
            track,
            n_turns, mu, barrier, kick_module, inverse
        );
    }

    std::vector<std::vector<double>> track_MEGNO(
        std::vector<unsigned int> n_turns, double mu,
        double barrier = 100.0,
        double kick_module = NAN,
        bool inverse = false) override
    {
        PYBIND11_OVERRIDE(
            std::vector<std::vector<double>>,
            cpu_henon,
            track_MEGNO,
            n_turns, mu, barrier, kick_module, inverse);
    }

    std::vector<double> get_x() const override { PYBIND11_OVERRIDE(std::vector<double>, cpu_henon, get_x); }
    std::vector<double> get_px() const override { PYBIND11_OVERRIDE(std::vector<double>, cpu_henon, get_px); }
    std::vector<double> get_y() const override { PYBIND11_OVERRIDE(std::vector<double>, cpu_henon, get_y); }
    std::vector<double> get_py() const override { PYBIND11_OVERRIDE(std::vector<double>, cpu_henon, get_py); }
    std::vector<unsigned int> get_steps() const override { PYBIND11_OVERRIDE(std::vector<unsigned int>, cpu_henon, get_steps); }

    void set_x(std::vector<double> x) override {PYBIND11_OVERRIDE(void, cpu_henon, set_x, x);}
    void set_px(std::vector<double> px) override {PYBIND11_OVERRIDE(void, cpu_henon, set_px, px);}
    void set_y(std::vector<double> y) override {PYBIND11_OVERRIDE(void, cpu_henon, set_y, y);}
    void set_py(std::vector<double> py) override {PYBIND11_OVERRIDE(void, cpu_henon, set_py, py);}
    void set_steps(std::vector<unsigned int> steps) override {PYBIND11_OVERRIDE(void, cpu_henon, set_steps, steps);}
    void set_steps(unsigned int unique_step) override {PYBIND11_OVERRIDE(void, cpu_henon, set_steps, unique_step);}
};

class pygpu_henon : public gpu_henon {
public:
    /* Inherit the constructors */
    using gpu_henon::gpu_henon;

    /* Trampoline (need one for each virtual function) */
    void reset() override {PYBIND11_OVERRIDE(void, gpu_henon, reset, );}
    
    void compute_a_modulation(unsigned int N, double omega_x, double omega_y, std::string modulation_kind = "sps", double omega_0 = NAN, double epsilon = 0.0, unsigned int offset = 0) override
    {
        PYBIND11_OVERRIDE(
            void, 
            gpu_henon, 
            compute_a_modulation, 
            N, omega_x, omega_y, modulation_kind, omega_0, epsilon, offset
        );
    }

    void track(unsigned int n_turns, double mu, double barrier = 100.0, double kick_module = NAN, bool inverse = false) override
    {
        PYBIND11_OVERRIDE(
            void,
            gpu_henon,
            track,
            n_turns, mu, barrier, kick_module, inverse
        );
    }

    std::vector<std::vector<double>> track_MEGNO(
        std::vector<unsigned int> n_turns, double mu,
        double barrier = 100.0,
        double kick_module = NAN,
        bool inverse = false) override
    {
        PYBIND11_OVERRIDE(
            std::vector<std::vector<double>>,
            gpu_henon,
            track_MEGNO,
            n_turns, mu, barrier, kick_module, inverse);
    }

    std::vector<double> get_x() const override { PYBIND11_OVERRIDE(std::vector<double>, gpu_henon, get_x); }
    std::vector<double> get_px() const override { PYBIND11_OVERRIDE(std::vector<double>, gpu_henon, get_px); }
    std::vector<double> get_y() const override { PYBIND11_OVERRIDE(std::vector<double>, gpu_henon, get_y); }
    std::vector<double> get_py() const override { PYBIND11_OVERRIDE(std::vector<double>, gpu_henon, get_py); }
    std::vector<unsigned int> get_steps() const override { PYBIND11_OVERRIDE(std::vector<unsigned int>, gpu_henon, get_steps); }

    void set_x(std::vector<double> x) override {PYBIND11_OVERRIDE(void, gpu_henon, set_x, x);}
    void set_px(std::vector<double> px) override {PYBIND11_OVERRIDE(void, gpu_henon, set_px, px);}
    void set_y(std::vector<double> y) override {PYBIND11_OVERRIDE(void, gpu_henon, set_y, y);}
    void set_py(std::vector<double> py) override {PYBIND11_OVERRIDE(void, gpu_henon, set_py, py);}
    void set_steps(std::vector<unsigned int> steps) override {PYBIND11_OVERRIDE(void, gpu_henon, set_steps, steps);}
    void set_steps(unsigned int unique_step) override {PYBIND11_OVERRIDE(void, gpu_henon, set_steps, unique_step);}
};


PYBIND11_MODULE(henon_map_engine, m)
{
    m.doc() = "Henon map engine";
    m.def("is_cuda_device_available", &is_cuda_device_available);

    m.def("basic_modulation", &basic_modulation, "Basic modulation",
        py::arg("tune"), py::arg("omega"), py::arg("epsilon"), py::arg("start"),
        py::arg("end")
    );
    m.def("sps_modulation", &sps_modulation, "SPS modulation",
        py::arg("tune"), py::arg("epsilon"), py::arg("start"), py::arg("end")
    );
    m.def("gaussian_modulation", &gaussian_modulation, "Gaussian modulation",
        py::arg("tune"), py::arg("sigma"), py::arg("start"), py::arg("end")
    );
    m.def("uniform_modulation", &uniform_modulation, "Uniform modulation",
        py::arg("from"), py::arg("to"), py::arg("start"), py::arg("end")
    );

    py::class_<henon, pyhenon>(m, "henon")
        .def(py::init<>())
        .def("reset", &henon::reset, "Resets the engine")

        .def("compute_a_modulation", &henon::compute_a_modulation, "Computes a modulation",
            py::arg("N"), py::arg("omega_x"), py::arg("omega_y"), py::arg("modulation_kind") = "sps", py::arg("omega_0") = NAN, py::arg("epsilon") = 0.0, py::arg("offset") = 0
        )

        .def("track", &henon::track, "Tracks the particles",
             py::arg("n_turns"), py::arg("mu"),
             py::arg("barrier") = 100.0, py::arg("kick_module") = NAN,
             py::arg("inverse") = false)

        .def("track_MEGNO", &henon::track_MEGNO,
             "Tracks the particles with MEGNO",
             py::arg("n_turns"),
             py::arg("mu"), py::arg("barrier") = 100.0,
             py::arg("kick_module") = NAN,
             py::arg("inverse") = false)

        .def("full_track", &henon::full_track, "Tracks the particles",
             py::arg("n_turns"), py::arg("mu"),
             py::arg("barrier") = 100.0, py::arg("kick_module") = NAN)

        .def("birkhoff_tunes", &henon::birkhoff_tunes, "Tracks the particles",
             py::arg("n_turns"), py::arg("mu"),
             py::arg("barrier") = 100.0, py::arg("kick_module") = NAN,
             py::arg("from") = std::vector<unsigned int>{0},
             py::arg("to") = std::vector<unsigned int>{0})

        .def("fft_tunes", &henon::fft_tunes, "Tracks the particles",
             py::arg("n_turns"), py::arg("mu"),
             py::arg("barrier") = 100.0, py::arg("kick_module") = NAN,
             py::arg("from") = std::vector<unsigned int>{0},
             py::arg("to") = std::vector<unsigned int>{0})

        .def("get_x", &henon::get_x, "Returns the x coordinates",
             py::return_value_policy::copy)
        .def("get_y", &henon::get_y, "Returns the y coordinates",
             py::return_value_policy::copy)
        .def("get_px", &henon::get_px, "Returns the x momenta",
             py::return_value_policy::copy)
        .def("get_py", &henon::get_py, "Returns the y momenta",
             py::return_value_policy::copy)

        .def("get_x0", &henon::get_x0, "Returns the initial x coordinates",
             py::return_value_policy::copy)
        .def("get_y0", &henon::get_y0, "Returns the initial y coordinates",
             py::return_value_policy::copy)
        .def("get_px0", &henon::get_px0, "Returns the initial x momenta",
             py::return_value_policy::copy)
        .def("get_py0", &henon::get_py0, "Returns the initial y momenta",
             py::return_value_policy::copy)

        .def("get_steps", &henon::get_steps,
             "Returns the number of steps before particle loss", py::return_value_policy::copy)

        .def("get_global_steps", &henon::get_global_steps,
             "Returns the number of global steps performed by the engine",
             py::return_value_policy::copy)

        .def("set_x", &henon::set_x,
             "Sets the x coordinates", py::arg("x"))
        .def("set_y", &henon::set_y,
             "Sets the y coordinates", py::arg("y"))
        .def("set_px", &henon::set_px,
             "Sets the x momenta", py::arg("px"))
        .def("set_py", &henon::set_py,
             "Sets the y momenta", py::arg("py"))

        .def("set_steps",
             static_cast<void (henon::*)(unsigned int)>(&henon::set_steps),
             "Sets the number of steps before particle loss", py::arg("steps"))
        .def("set_steps",
             static_cast<void (henon::*)(std::vector<unsigned int>)>(&henon::set_steps),
             "Sets the number of steps before particle loss", py::arg("steps"))
        .def("set_global_steps", &henon::set_global_steps,
             "Sets the number of global steps performed by the engine",
             py::arg("global_steps"));

    py::class_<gpu_henon, henon, pygpu_henon>(m, "gpu_henon")
        .def(py::init<
                 const std::vector<double> &,
                 const std::vector<double> &,
                 const std::vector<double> &,
                 const std::vector<double> &>(),
             py::arg("x_init"), py::arg("px_init"), py::arg("y_init"),
             py::arg("py_init"));

    py::class_<cpu_henon, henon, pycpu_henon>(m, "cpu_henon")
        .def(py::init<
                 const std::vector<double> &,
                 const std::vector<double> &,
                 const std::vector<double> &,
                 const std::vector<double> &>(),
             py::arg("x_init"), py::arg("px_init"), py::arg("y_init"),
             py::arg("py_init"));
}
