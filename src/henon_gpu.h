#ifndef HENON_GPU_
#define HENON_GPU_

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "utils.h"
#include "henon.h"

#ifdef USE_CUDA

struct particles_4d_gpu : public particles_4d {
  double *d_x;
  double *d_px;
  double *d_y;
  double *d_py;

  int *d_steps;
  uint8_t *d_valid;
  uint8_t *d_ghost;

  size_t *d_idx;
  size_t *d_idx_base;

  curandState *d_rng_state;

  void _general_cudaMalloc();
  void _general_cudaFree();
  void _general_host_to_device_copy();
  void _general_device_to_host_copy();

  particles_4d_gpu() = default;

  particles_4d_gpu(const std::vector<double> &x_,
                   const std::vector<double> &px_,
                   const std::vector<double> &y_,
                   const std::vector<double> &py_);

  virtual void reset() override;
  virtual void add_ghost(const double &displacement_module,
                         const std::string &direction) override;

  virtual void renormalize(const double &module_target) override;

  virtual const std::vector<std::vector<double>>
  get_displacement_module() const override;
  virtual const std::vector<std::vector<std::vector<double>>>
  get_displacement_direction() const override;

  virtual const std::vector<double> get_x() const override;
  virtual const std::vector<double> get_px() const override;
  virtual const std::vector<double> get_y() const override;
  virtual const std::vector<double> get_py() const override;
  virtual const std::vector<int> get_steps() const override;

  virtual std::vector<double> get_radius() const override;
  virtual double get_radius_mean() const override;
  virtual double get_radius_std() const override;
  virtual std::vector<double> get_action() const override;
  virtual double get_action_mean() const override;
  virtual double get_action_std() const override;
  virtual std::vector<double> get_action_x() const override;
  virtual double get_action_x_mean() const override;
  virtual double get_action_x_std() const override;
  virtual std::vector<double> get_action_y() const override;
  virtual double get_action_y_mean() const override;
  virtual double get_action_y_std() const override;
  virtual std::vector<double> get_angles_x() const override;
  virtual double get_angles_x_mean() const override;
  virtual double get_angles_x_std() const override;
  virtual std::vector<double> get_angles_y() const override;
  virtual double get_angles_y_mean() const override;
  virtual double get_angles_y_std() const override;

  virtual const std::vector<uint8_t> get_valid() const override;
  virtual const std::vector<uint8_t> get_ghost() const override;

  virtual const std::vector<size_t> get_idx() const override;
  virtual const std::vector<size_t> get_idx_base() const override;

  virtual const size_t &get_n_particles() const override;
  virtual const size_t &get_n_ghosts_per_particle() const override;

  virtual ~particles_4d_gpu();
  size_t _optimal_nblocks() const;
};

class henon_tracker_gpu : public henon_tracker {
public:
  double *d_omega_x_sin;
  double *d_omega_x_cos;
  double *d_omega_y_sin;
  double *d_omega_y_cos;

  curandState *d_rand_states;

  henon_tracker_gpu() = default;
  henon_tracker_gpu(unsigned int N, double omega_x, double omega_y,
                    std::string modulation_kind = "sps", double omega_0 = NAN,
                    double epsilon = 0.0, int offset = 0);
  ~henon_tracker_gpu();

  virtual void compute_a_modulation(unsigned int N, double omega_x,
                                    double omega_y,
                                    std::string modulation_kind = "sps",
                                    double omega_0 = NAN, double epsilon = 0.0,
                                    int offset = 0) override;

  void track(particles_4d_gpu &particles, unsigned int n_turns, double mu,
             double barrier = 100.0, double kick_module = NAN,
             bool inverse = false);

  std::vector<std::vector<std::vector<double>>>
  get_tangent_matrix(const particles_4d_gpu &particles, const double &mu,
                     const bool &reverse) const;
};

struct matrix_4d_vector_gpu {
  double *d_matrix;
  size_t N;
  size_t n_blocks;

  matrix_4d_vector_gpu(size_t _N);
  ~matrix_4d_vector_gpu();

  void reset();
  void structured_multiply(const henon_tracker_gpu &tracker,
                           const particles_4d_gpu &particles, const double &mu);
  void set_with_tracker(const henon_tracker_gpu &tracker,
                        const particles_4d_gpu &particles, const double &mu);
  void explicit_copy(const matrix_4d_vector_gpu &matrix);

  const std::vector<std::vector<std::vector<double>>> get_matrix() const;
  std::vector<std::vector<double>>
  get_vector(const std::vector<std::vector<double>> &rv) const;
};

struct vector_4d_gpu {
  double *d_vectors;
  size_t N;
  size_t n_blocks;

  vector_4d_gpu(const std::vector<std::vector<double>> &rv);
  ~vector_4d_gpu();

  void set_vectors(const std::vector<std::vector<double>> &rv);
  void set_vectors(const std::vector<double> &rv);
  void multiply(const matrix_4d_vector_gpu &matrix);
  void normalize();

  const std::vector<std::vector<double>> get_vectors() const;
};

struct lyapunov_birkhoff_construct {
  double *d_vector;
  double *d_vector_b;
  double *d_birkhoff;
  size_t N;
  size_t n_blocks;
  size_t n_weights;
  size_t idx;

  lyapunov_birkhoff_construct(size_t _N, size_t _n_weights);
  ~lyapunov_birkhoff_construct();

  void reset();
  void change_weights(size_t _n_weights);
  void add(const vector_4d_gpu &vectors);

  std::vector<double> get_weights() const;
  std::vector<double> get_values_raw() const;
  std::vector<double> get_values_b() const;
};

struct lyapunov_birkhoff_construct_multi {
  std::vector<double *> d_vector;
  std::vector<double *> d_vector_b;
  std::vector<double *> d_birkhoff;
  size_t N;
  size_t n_blocks;
  std::vector<size_t> n_weights;
  size_t idx;

  lyapunov_birkhoff_construct_multi(size_t _N, std::vector<size_t> _n_weights);
  ~lyapunov_birkhoff_construct_multi();

  void reset();
  void add(const vector_4d_gpu &vectors);

  std::vector<std::vector<double>> get_weights() const;
  std::vector<std::vector<double>> get_values_raw() const;
  std::vector<std::vector<double>> get_values_b() const;
};

struct megno_construct {
  double *d_layer1;
  double *d_layer2;
  double *d_layer3;
  size_t N;
  size_t n_blocks;
  size_t idx;

  megno_construct(size_t _N);
  ~megno_construct();

  void reset();
  void add(const matrix_4d_vector_gpu &matrix_a,
           const matrix_4d_vector_gpu &matrix_b);

  std::vector<double> get_values() const;
};

struct megno_birkhoff_construct {
  double *d_layer1;
  double **d_layer2;
  double **d_weights;
  size_t *d_turn_samples;
  size_t n_turn_samples;
  size_t N;
  size_t n_blocks;
  size_t idx;

  megno_birkhoff_construct(size_t _N, std::vector<size_t> _turn_samples);
  ~megno_birkhoff_construct();

  void reset();
  void add(const matrix_4d_vector_gpu &matrix_a,
           const matrix_4d_vector_gpu &matrix_b);

  std::vector<std::vector<double>> get_values() const;
};

struct tune_birkhoff_construct {
  double **d_tune1_x;
  double **d_tune1_y;
  double **d_tune2_x;
  double **d_tune2_y;
  double **d_weights;
  double *store_x;
  double *store_y;
  size_t *d_turn_samples;
  size_t n_turn_samples;
  size_t N;
  size_t n_blocks;
  size_t idx;

  tune_birkhoff_construct(size_t _N, std::vector<size_t> _turn_samples);
  ~tune_birkhoff_construct();

  void reset();
  void first_add(const particles_4d_gpu &particles);
  void add(const particles_4d_gpu &particles);

  std::vector<std::vector<double>> get_tune1_x() const;
  std::vector<std::vector<double>> get_tune1_y() const;
  std::vector<std::vector<double>> get_tune2_x() const;
  std::vector<std::vector<double>> get_tune2_y() const;
};

struct storage_4d_gpu {
  size_t N;
  size_t batch_size;
  size_t n_blocks;
  size_t idx;
  double *d_x;
  double *d_px;
  double *d_y;
  double *d_py;

  storage_4d_gpu(size_t _N, size_t _batch_size);
  ~storage_4d_gpu();

  void store(const particles_4d_gpu &particles);
  void reset();

  const std::vector<std::vector<double>> get_x() const;
  const std::vector<std::vector<double>> get_px() const;
  const std::vector<std::vector<double>> get_y() const;
  const std::vector<std::vector<double>> get_py() const;
};

#endif // USE_CUDA

#endif // HENON_GPU_