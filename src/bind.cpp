#include <vector>
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
#include "dynamic_indicator.h"

bool has_cuda_error_happened()
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return true;
    }
    return false;
}

bool is_cuda_device_available()
{
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

struct py_particles_4d : public particles_4d
{
public:
    /* Inherit the constructors */
    using particles_4d::particles_4d;

    void reset() override { PYBIND11_OVERRIDE(void, particles_4d, reset, ); }
    void add_ghost(const double &displacement_module, const std::string &direction) override
    {
        PYBIND11_OVERRIDE(void, particles_4d, add_ghost, displacement_module, direction);
    }

    void renormalize(const double &module_target) override
    {
        PYBIND11_OVERRIDE(void, particles_4d, renormalize, module_target);
    }

    const std::vector<std::vector<double>> get_displacement_module() const override
    {
        PYBIND11_OVERRIDE(const std::vector<std::vector<double>>, particles_4d, get_displacement_module, );
    }
    const std::vector<std::vector<std::vector<double>>> get_displacement_direction() const override
    {
        PYBIND11_OVERRIDE(const std::vector<std::vector<std::vector<double>>>, particles_4d, get_displacement_direction);
    }

    const std::vector<double> get_x() const override
    {
        PYBIND11_OVERRIDE(const std::vector<double>, particles_4d, get_x);
    }
    const std::vector<double> get_px() const override
    {
        PYBIND11_OVERRIDE(const std::vector<double>, particles_4d, get_px);
    }
    const std::vector<double> get_y() const override
    {
        PYBIND11_OVERRIDE(const std::vector<double>, particles_4d, get_y);
    }
    const std::vector<double> get_py() const override
    {
        PYBIND11_OVERRIDE(const std::vector<double>, particles_4d, get_py);
    }
    const std::vector<unsigned int> get_steps() const override
    {
        PYBIND11_OVERRIDE(const std::vector<unsigned int>, particles_4d, get_steps);
    }

    std::vector<double> get_radius() const override
    {
        PYBIND11_OVERRIDE(const std::vector<double>, particles_4d, get_radius);
    }
    double get_radius_mean() const override
    {
        PYBIND11_OVERRIDE(const double, particles_4d, get_radius_mean);
    }
    double get_radius_std() const override
    {
        PYBIND11_OVERRIDE(const double, particles_4d, get_radius_std);
    }

    std::vector<double> get_action() const override
    {
        PYBIND11_OVERRIDE(const std::vector<double>, particles_4d, get_action);
    }
    double get_action_mean() const override
    {
        PYBIND11_OVERRIDE(const double, particles_4d, get_action_mean);
    }
    double get_action_std() const override
    {
        PYBIND11_OVERRIDE(const double, particles_4d, get_action_std);
    }

    std::vector<double> get_action_x() const override
    {
        PYBIND11_OVERRIDE(const std::vector<double>, particles_4d, get_action_x);
    }
    double get_action_x_mean() const override
    {
        PYBIND11_OVERRIDE(const double, particles_4d, get_action_x_mean);
    }
    double get_action_x_std() const override
    {
        PYBIND11_OVERRIDE(const double, particles_4d, get_action_x_std);
    }

    std::vector<double> get_action_y() const override
    {
        PYBIND11_OVERRIDE(const std::vector<double>, particles_4d, get_action_y);
    }
    double get_action_y_mean() const override
    {
        PYBIND11_OVERRIDE(const double, particles_4d, get_action_y_mean);
    }
    double get_action_y_std() const override
    {
        PYBIND11_OVERRIDE(const double, particles_4d, get_action_y_std);
    }

    std::vector<double> get_angles_x() const override
    {
        PYBIND11_OVERRIDE(const std::vector<double>, particles_4d, get_angles_x);
    }
    double get_angles_x_mean() const override
    {
        PYBIND11_OVERRIDE(const double, particles_4d, get_angles_x_mean);
    }
    double get_angles_x_std() const override
    {
        PYBIND11_OVERRIDE(const double, particles_4d, get_angles_x_std);
    }

    std::vector<double> get_angles_y() const override
    {
        PYBIND11_OVERRIDE(const std::vector<double>, particles_4d, get_angles_y);
    }
    double get_angles_y_mean() const override
    {
        PYBIND11_OVERRIDE(const double, particles_4d, get_angles_y_mean);
    }
    double get_angles_y_std() const override
    {
        PYBIND11_OVERRIDE(const double, particles_4d, get_angles_y_std);
    }

    const std::vector<uint8_t> get_valid() const override
    {
        PYBIND11_OVERRIDE(const std::vector<uint8_t>, particles_4d, get_valid);
    }
    const std::vector<uint8_t> get_ghost() const override
    {
        PYBIND11_OVERRIDE(const std::vector<uint8_t>, particles_4d, get_ghost);
    }

    const std::vector<size_t> get_idx() const override
    {
        PYBIND11_OVERRIDE(const std::vector<size_t>, particles_4d, get_idx);
    }
    const std::vector<size_t> get_idx_base() const override
    {
        PYBIND11_OVERRIDE(const std::vector<size_t>, particles_4d, get_idx_base);
    }

    const size_t &get_n_particles() const override
    {
        PYBIND11_OVERRIDE(const size_t &, particles_4d, get_n_particles);
    }
    const size_t &get_n_ghosts_per_particle() const override
    {
        PYBIND11_OVERRIDE(const size_t &, particles_4d, get_n_ghosts_per_particle);
    }
};

struct py_henon_tracker : public henon_tracker
{
    using henon_tracker::henon_tracker;

    void compute_a_modulation(unsigned int N, double omega_x, double omega_y, std::string modulation_kind = "sps", double omega_0 = NAN, double epsilon = 0.0, unsigned int offset = 0) override
    {
        PYBIND11_OVERRIDE(
            void,
            henon_tracker,
            compute_a_modulation,
            N, omega_x, omega_y, modulation_kind, omega_0, epsilon, offset);
    }
};

struct py_henon_tracker_gpu : public henon_tracker_gpu
{
    using henon_tracker_gpu::henon_tracker_gpu;

    void compute_a_modulation(unsigned int N, double omega_x, double omega_y, std::string modulation_kind = "sps", double omega_0 = NAN, double epsilon = 0.0, unsigned int offset = 0) override
    {
        PYBIND11_OVERRIDE(
            void,
            henon_tracker_gpu,
            compute_a_modulation,
            N, omega_x, omega_y, modulation_kind, omega_0, epsilon, offset);
    }
};

PYBIND11_MODULE(henon_map_engine, m)
{
    m.doc() = "Henon map engine";
    m.def("is_cuda_device_available", &is_cuda_device_available);

    m.def("basic_modulation", &basic_modulation, "Basic modulation",
          py::arg("tune"), py::arg("omega"), py::arg("epsilon"), py::arg("start"),
          py::arg("end"));
    m.def("sps_modulation", &sps_modulation, "SPS modulation",
          py::arg("tune"), py::arg("epsilon"), py::arg("start"), py::arg("end"));
    m.def("gaussian_modulation", &gaussian_modulation, "Gaussian modulation",
          py::arg("tune"), py::arg("sigma"), py::arg("start"), py::arg("end"));
    m.def("uniform_modulation", &uniform_modulation, "Uniform modulation",
          py::arg("from"), py::arg("to"), py::arg("start"), py::arg("end"));

    m.def("birkhoff_weights", &birkhoff_weights, "Birkhoff weights", py::arg("n_weights"));

    m.def("get_tunes", &get_tunes, "Get tunes",
          py::arg("x"), py::arg("px"));

    py::class_<particles_4d, py_particles_4d>(m, "particles_4d")
        .def(py::init<
             const std::vector<double> &,
             const std::vector<double> &,
             const std::vector<double> &,
             const std::vector<double> &>())
        .def("reset", &particles_4d::reset)
        .def("add_ghost", &particles_4d::add_ghost,
             py::arg("displacement_module"), py::arg("direction"))
        .def("renormalize", &particles_4d::renormalize, py::arg("module_target"))
        .def("get_displacement_module", &particles_4d::get_displacement_module)
        .def("get_displacement_direction", &particles_4d::get_displacement_direction)
        .def("get_x", [](particles_4d &self)
             {
            py::array out = py::cast(self.get_x());
            return out; })
        .def("get_px", [](particles_4d &self)
             {
            py::array out = py::cast(self.get_px());
            return out; })
        .def("get_y", [](particles_4d &self)
             {
            py::array out = py::cast(self.get_y());
            return out; })
        .def("get_py", [](particles_4d &self)
             {
            py::array out = py::cast(self.get_py());
            return out; })
        .def("get_steps", [](particles_4d &self)
             {
            py::array out = py::cast(self.get_steps());
            return out; })
        .def("get_radius", [](particles_4d &self)
             {
            py::array out = py::cast(self.get_radius());
            return out; })
        .def("get_radius_mean", [](particles_4d &self)
             {
            py::array out = py::cast(self.get_radius_mean());
            return out; })
        .def("get_radius_std", [](particles_4d &self)
             {
            py::array out = py::cast(self.get_radius_std());
            return out; })
        .def("get_action", [](particles_4d &self)
             {
            py::array out = py::cast(self.get_action());
            return out; })
        .def("get_action_mean", [](particles_4d &self)
             {
            py::array out = py::cast(self.get_action_mean());
            return out; })
        .def("get_action_std", [](particles_4d &self)
             {
            py::array out = py::cast(self.get_action_std());
            return out; })
        .def("get_action_x", [](particles_4d &self)
             {
            py::array out = py::cast(self.get_action_x());
            return out; })
        .def("get_action_x_mean", [](particles_4d &self)
             {
            py::array out = py::cast(self.get_action_x_mean());
            return out; })
        .def("get_action_x_std", [](particles_4d &self)
             {
            py::array out = py::cast(self.get_action_x_std());
            return out; })
        .def("get_action_y", [](particles_4d &self)
             {
            py::array out = py::cast(self.get_action_y());
            return out; })
        .def("get_action_y_mean", [](particles_4d &self)
             {
            py::array out = py::cast(self.get_action_y_mean());
            return out; })
        .def("get_action_y_std", [](particles_4d &self)
             {
            py::array out = py::cast(self.get_action_y_std());
            return out; })
        .def("get_angles_x", [](particles_4d &self)
             {
            py::array out = py::cast(self.get_angles_x());
            return out; })
        .def("get_angles_x_mean", [](particles_4d &self)
             {
            py::array out = py::cast(self.get_angles_x_mean());
            return out; })
        .def("get_angles_x_std", [](particles_4d &self)
             {
            py::array out = py::cast(self.get_angles_x_std());
            return out; })
        .def("get_angles_y", [](particles_4d &self)
             {
            py::array out = py::cast(self.get_angles_y());
            return out; })
        .def("get_angles_y_mean", [](particles_4d &self)
             {
            py::array out = py::cast(self.get_angles_y_mean());
            return out; })
        .def("get_angles_y_std", [](particles_4d &self)
             {
            py::array out = py::cast(self.get_angles_y_std());
            return out; })
        .def("get_valid", [](particles_4d &self)
             {
            py::array out = py::cast(self.get_valid());
            return out; })
        .def("get_ghost", [](particles_4d &self)
             {
            py::array out = py::cast(self.get_ghost());
            return out; })
        .def("get_idx", [](particles_4d &self)
             {
            py::array out = py::cast(self.get_idx());
            return out; })
        .def("get_idx_base", [](particles_4d &self)
             {
            py::array out = py::cast(self.get_idx_base());
            return out; })
        .def("get_n_particles", [](particles_4d &self)
             {
            py::array out = py::cast(self.get_n_particles());
            return out; })
        .def("get_n_ghosts_per_particle", [](particles_4d &self)
             {
            py::array out = py::cast(self.get_n_ghosts_per_particle());
            return out; });

    py::class_<matrix_4d_vector>(m, "matrix_4d_vector")
        .def(py::init<size_t>())
        .def("reset", &matrix_4d_vector::reset)
        .def("multiply", &matrix_4d_vector::multiply, py::arg("matrix"))
        .def("structured_multiply",
             static_cast<void (matrix_4d_vector::*)(const henon_tracker &, const particles_4d &, const double &, const bool &)>(&matrix_4d_vector::structured_multiply),
             py::arg("tracker"), py::arg("particles"), py::arg("mu"), py::arg("reverse"))
        .def("structured_multiply",
             static_cast<void (matrix_4d_vector::*)(const henon_tracker_gpu &, const particles_4d_gpu &, const double &, const bool &)>(&matrix_4d_vector::structured_multiply),
             py::arg("tracker"), py::arg("particles"), py::arg("mu"), py::arg("reverse"))
        .def("get_matrix", [](matrix_4d_vector &self)
             {
            py::array out = py::cast(self.get_matrix());
            return out; })
        .def(
            "get_vector", [](matrix_4d_vector &self, const std::vector<std::vector<double>> &vector)
            {
            py::array out = py::cast(self.get_vector(vector));
            return out; },
            py::arg("vector"));

    py::class_<matrix_4d_vector_gpu>(m, "matrix_4d_vector_gpu")
        .def(py::init<size_t>())
        .def("reset", &matrix_4d_vector_gpu::reset)
        .def("structured_multiply", &matrix_4d_vector_gpu::structured_multiply, py::arg("tracker"), py::arg("particles"), py::arg("mu"))
        .def("set_with_tracker", &matrix_4d_vector_gpu::set_with_tracker, py::arg("tracker"), py::arg("particles"), py::arg("mu"))
        .def("explicit_copy", &matrix_4d_vector_gpu::explicit_copy, py::arg("other"))
        .def("get_matrix", [](matrix_4d_vector_gpu &self)
             {
            py::array out = py::cast(self.get_matrix());
            return out; })
        .def(
            "get_vector", [](matrix_4d_vector_gpu &self, const std::vector<std::vector<double>> &vector)
            {
            py::array out = py::cast(self.get_vector(vector));
            return out; },
            py::arg("vector"));

    py::class_<vector_4d_gpu>(m, "vector_4d_gpu")
        .def(py::init<const std::vector<std::vector<double>> &>())
        .def("set_vectors", static_cast<void (vector_4d_gpu::*)(const std::vector<std::vector<double>> &)>(&vector_4d_gpu::set_vectors), py::arg("vectors"))
        .def("set_vectors", static_cast<void (vector_4d_gpu::*)(const std::vector<double> &)>(&vector_4d_gpu::set_vectors), py::arg("vectors"))
        .def("multiply", &vector_4d_gpu::multiply, py::arg("matrix"))
        .def("normalize", &vector_4d_gpu::normalize)
        .def("get_vectors", [](vector_4d_gpu &self)
             {
            py::array out = py::cast(self.get_vectors());
            return out; });

    py::class_<lyapunov_birkhoff_construct>(m, "lyapunov_birkhoff_construct")
        .def(py::init<size_t, size_t>())
        .def("reset", &lyapunov_birkhoff_construct::reset)
        .def("change_weights", &lyapunov_birkhoff_construct::change_weights, py::arg("n_weights"))
        .def("add", &lyapunov_birkhoff_construct::add, py::arg("vectors"))
        .def("get_weights", [](lyapunov_birkhoff_construct &self)
             {
            py::array out = py::cast(self.get_weights());
            return out; })
        .def("get_values_raw", [](lyapunov_birkhoff_construct &self)
             {
            py::array out = py::cast(self.get_values_raw());
            return out; })
        .def("get_values_b", [](lyapunov_birkhoff_construct &self)
             {
            py::array out = py::cast(self.get_values_b());
            return out; });

    py::class_<lyapunov_birkhoff_construct_multi>(m, "lyapunov_birkhoff_construct_multi")
        .def(py::init<size_t, std::vector<size_t>>())
        .def("reset", &lyapunov_birkhoff_construct_multi::reset)
        .def("add", &lyapunov_birkhoff_construct_multi::add, py::arg("vectors"))
        .def("get_values_raw", [](lyapunov_birkhoff_construct_multi &self)
             {
            py::array out = py::cast(self.get_values_raw());
            return out; })
        .def("get_values_b", [](lyapunov_birkhoff_construct_multi &self)
             {
            py::array out = py::cast(self.get_values_b());
            return out; });

    py::class_<megno_construct>(m, "megno_construct")
        .def(py::init<size_t>())
        .def("reset", &megno_construct::reset)
        .def("add", &megno_construct::add, py::arg("matrix_a"), py::arg("matrix_b"))
        .def("get_values", [](megno_construct &self)
             {
            py::array out = py::cast(self.get_values());
            return out; });

    py::class_<megno_birkhoff_construct>(m, "megno_birkhoff_construct")
        .def(py::init<size_t, std::vector<size_t>>())
        .def("reset", &megno_birkhoff_construct::reset)
        .def("add", &megno_birkhoff_construct::add, py::arg("matrix_a"), py::arg("matrix_b"))
        .def("get_values", [](megno_birkhoff_construct &self)
             {
            py::array out = py::cast(self.get_values());
            return out; });

    py::class_<tune_birkhoff_construct>(m, "tune_birkhoff_construct")
        .def(py::init<size_t, std::vector<size_t>>())
        .def("reset", &tune_birkhoff_construct::reset)
        .def("first_add", &tune_birkhoff_construct::first_add, py::arg("particles"))
        .def("add", &tune_birkhoff_construct::add, py::arg("particles"))
        .def("get_tune1_x", [](tune_birkhoff_construct &self)
             {
            py::array out = py::cast(self.get_tune1_x());
            return out; })
        .def("get_tune1_y", [](tune_birkhoff_construct &self)
             {
            py::array out = py::cast(self.get_tune1_y());
            return out; })
        .def("get_tune2_x", [](tune_birkhoff_construct &self)
             {
            py::array out = py::cast(self.get_tune2_x());
            return out; })
        .def("get_tune2_y", [](tune_birkhoff_construct &self)
             {
            py::array out = py::cast(self.get_tune2_y());
            return out; });

    py::class_<particles_4d_gpu, particles_4d>(m, "particles_4d_gpu")
        .def(py::init<
             const std::vector<double> &,
             const std::vector<double> &,
             const std::vector<double> &,
             const std::vector<double> &>());

    py::class_<henon_tracker, py_henon_tracker>(m, "henon_tracker")
        .def(py::init<
             unsigned int,
             double,
             double,
             std::string,
             double,
             double,
             unsigned int>())
        .def("compute_a_modulation", &henon_tracker::compute_a_modulation,
             "Compute a modulation",
             py::arg("N"), py::arg("omega_x"), py::arg("omega_y"), py::arg("modulation_kind"), py::arg("omega_0"), py::arg("epsilon"), py::arg("offset"))
        .def("track", &henon_tracker::track, "Track particles", py::arg("particles"), py::arg("n_turns"), py::arg("mu"), py::arg("barrier") = 100.0, py::arg("kick_module") = NAN, py::arg("inverse") = false)
        .def("megno", &henon_tracker::megno, "Compute megno", py::arg("particles"), py::arg("n_turns"), py::arg("mu"), py::arg("barrier") = 100.0, py::arg("kick_module") = NAN, py::arg("inverse") = false, py::arg("turn_samples") = std::vector<unsigned int>(), py::arg("n_threads") = -1)
        .def("birkhoff_tunes", &henon_tracker::birkhoff_tunes, "Compute birkhoff tunes", py::arg("particles"), py::arg("n_turns"), py::arg("mu"), py::arg("barrier") = 100.0, py::arg("kick_module") = NAN, py::arg("inverse") = false, py::arg("from_idx") = std::vector<unsigned int>(), py::arg("to_idx") = std::vector<unsigned int>())
        .def("all_tunes", &henon_tracker::all_tunes, "Compute birkhoff tunes", py::arg("particles"), py::arg("n_turns"), py::arg("mu"), py::arg("barrier") = 100.0, py::arg("kick_module") = NAN, py::arg("inverse") = false, py::arg("from_idx") = std::vector<unsigned int>(), py::arg("to_idx") = std::vector<unsigned int>())
        .def(
            "get_tangent_matrix", [](henon_tracker &self, const particles_4d &particles, const double &mu, const bool &reverse)
            {
            py::array out = py::cast(self.get_tangent_matrix(particles, mu, reverse));
            return out; },
            "Get tangent matrix", py::arg("particles"), py::arg("mu"), py::arg("reverse"));

    py::class_<henon_tracker_gpu, py_henon_tracker_gpu>(m, "henon_tracker_gpu")
        .def(py::init<
             unsigned int,
             double,
             double,
             std::string,
             double,
             double,
             unsigned int>())
        .def("compute_a_modulation", &henon_tracker_gpu::compute_a_modulation,
             "Compute a modulation",
             py::arg("N"), py::arg("omega_x"), py::arg("omega_y"), py::arg("modulation_kind"), py::arg("omega_0"), py::arg("epsilon"), py::arg("offset"))
        .def("track", &henon_tracker_gpu::track, "Track particles", py::arg("particles"), py::arg("n_turns"), py::arg("mu"), py::arg("barrier") = 100.0, py::arg("kick_module") = NAN, py::arg("inverse") = false)
        .def(
            "get_tangent_matrix", [](henon_tracker_gpu &self, const particles_4d_gpu &particles, const double &mu, const bool &reverse)
            {
            py::array out = py::cast(self.get_tangent_matrix(particles, mu, reverse));
            return out; },
            "Get tangent matrix", py::arg("particles"), py::arg("mu"), py::arg("reverse"));

    py::class_<storage_4d>(m, "storage_4d")
        .def(py::init<size_t>())
        .def("store",
             static_cast<void (storage_4d::*)(const particles_4d &)>(&storage_4d::store),
             "Store particles")
        .def("store",
             static_cast<void (storage_4d::*)(const particles_4d_gpu &)>(&storage_4d::store),
             "Store particles")
        .def("tune_fft", &storage_4d::tune_fft, "Tune FFT", py::arg("from_idx"), py::arg("to_idx"))
        .def("tune_birkhoff", &storage_4d::tune_birkhoff, "Tune birkhoff", py::arg("from_idx"), py::arg("to_idx"))
        .def("get_x", &storage_4d::get_x)
        .def("get_px", &storage_4d::get_px)
        .def("get_y", &storage_4d::get_y)
        .def("get_py", &storage_4d::get_py);

    py::class_<storage_4d_gpu>(m, "storage_4d_gpu")
        .def(py::init<size_t, size_t>())
        .def("store", &storage_4d_gpu::store, "Store particles", py::arg("particles"))
        .def("reset", &storage_4d_gpu::reset, "Reset storage")
        .def("get_x", [](storage_4d_gpu &self)
             {
            py::array out = py::cast(self.get_x());
            return out; })
        .def("get_px", [](storage_4d_gpu &self)
             {
            py::array out = py::cast(self.get_px());
            return out; })
        .def("get_y", [](storage_4d_gpu &self)
             {
            py::array out = py::cast(self.get_y());
            return out; })
        .def("get_py", [](storage_4d_gpu &self)
             {
            py::array out = py::cast(self.get_py());
            return out; });
}