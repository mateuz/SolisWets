#ifndef _SOLISWETS_H
#define _SOLISWETS_H

#include "helper.cuh"

class SolisWets
{
public:
	unsigned int n_success;
	unsigned int n_fails;
	unsigned int n_dim;

	/* bounds */
	float x_min;
	float x_max;
	float delta;

	/* device datas */
	float * d_new_solution;
	float * d_bias;
	float * d_diff;
	curandState * d_states;

	unsigned int n_blocks;
	unsigned int n_threads;

  /*
   * @params:
   *    - unsigned int: n_success
   *    - unsigned int: n_fails
   *    - unsigned int: n_dim
	 *    - unsigned int: n_threads to execute the kernels
	 *    - unsigned int: n_blocks to execute the kernels
   *    - float: x_min (lower bound)
   *    - float: x_max (upper bound)
   *    - float: delta
   */
	SolisWets(unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, float, float, float);
	~SolisWets();

  float optimize(const unsigned int, float *);
};

__global__ void setup_kernel
(
	curandState * state,
	float * d_bias,
	unsigned int seed,
	unsigned int size
);

__global__ void increment_bias(
	float * d_bias,
	float * d_diff,
	unsigned int ndim
);

__global__ void decrement_bias(
	float * d_bias,
	float * d_diff,
	unsigned int ndim
);

__global__ void generate_new_solution(
	float * d_sol,
	float * d_new_solution,
	float * d_bias,
	float * d_diff,
	float x_min,
	float x_max,
	float delta,
	unsigned int n_dim,
	unsigned short direction,
	curandState * g_state
);

#endif
