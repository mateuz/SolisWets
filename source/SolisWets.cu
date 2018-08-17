#include "SolisWets.h"
#include <random>
#include <cassert>

SolisWets::SolisWets(
  unsigned int _n_success,
  unsigned int _n_fails,
  unsigned int _n_dim,
  unsigned int _n_threads,
  unsigned int _n_blocks,
  float _x_min,
  float _x_max,
  float _delta
){
  n_success = _n_success;
  n_fails   = _n_fails;
  n_dim     = _n_dim;
  n_threads = _n_threads;
  n_blocks  = _n_blocks;
  x_min     = _x_min;
  x_max     = _x_max;
  delta     = _delta;

  checkCudaErrors(cudaMalloc((void **)&d_new_solution, n_dim * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_bias, n_dim * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_diff, n_dim * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_states, n_dim * sizeof(curandStateXORWOW_t)));

  std::random_device rd;
  unsigned int seed = rd();

  assert(n_blocks  >= 1);
  assert(n_threads >= 1);
  assert((n_blocks * n_threads) >= n_dim);

  //setup the random generator
  setup_kernel<<<n_blocks, n_threads>>>(d_states, d_bias, seed, n_dim);
}

SolisWets::~SolisWets()
{
  checkCudaErrors(cudaFree(d_new_solution));
  checkCudaErrors(cudaFree(d_bias));
  checkCudaErrors(cudaFree(d_diff));
  checkCudaErrors(cudaFree(d_states));
}

float SolisWets::optimize(const unsigned int n_evals, float * d_sol){
  unsigned int _n_success = 0, _n_fails = 0;

  float result = 0.0 current_fitness = 0.0;

  //eval solution
  //current_fitness =

  for( unsigned int it = 1; it <= n_evals; ){
    generate_new_solution<<<n_blocks, n_threads>>>(
      d_sol, d_new_solution, d_bias,
      d_diff, x_min, x_max, delta,
      n_dim, 0, d_states);

    //calc fitness
    //result =
    it++;

    if( result < current_fitness ){

      current_fitness = result;

      //increment bias and copy
      increment_bias<<<n_blocks, n_threads>>>(d_bias, d_diff, n_dim);
      checkCudaErrors(cudaMemcpy(d_sol, d_new_solution, n_dim * sizeof(float)));

      _n_success++;
      _n_fails = 0;

    } else {
      if( it >= n_evals ) break;

      generate_new_solution<<<n_blocks, n_threads>>>(
        d_sol, d_new_solution, d_bias,
        d_diff, x_min, x_max, delta,
        n_dim, 1, d_states);

      //calc fitness
      //result =
      it++;

      if( result <= current_fitness ){
        current_fitness = result;

        //decrement bias and copy
        decrement_bias<<<n_blocks, n_threads>>>(d_bias, d_diff, n_dim);
        checkCudaErrors(cudaMemcpy(d_sol, d_new_solution, n_dim * sizeof(float)));

        _n_success++;
        _n_fails = 0;
      } else {
        _n_success = 0;
        _n_fails++;
      }
    }
    if( _n_success > n_success ){
      delta *= 2.0;
      _n_success = 0;
    } else if( _n_fails > n_fails ){
      delta /= 2;
      _n_fails = 0;
    }
  }
  return current_fitness;
}

__global__ void setup_kernel
(
  curandState * state,
  float * d_bias,
  unsigned int seed,
  unsigned int size
){
  unsigned int index = threadIdx.x + (blockIdx.x * blockDim.x);

  /* Each thread gets same seed, a different sequence number, no offset */
  if (index < size) {
    curand_init(seed, index, 0, &state[index]);
    d_bias[index] = 0.0;
  }
}

__global__ void increment_bias(float * d_bias, float * d_diff, unsigned int ndim)
{
  unsigned int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if( index < ndim ){
    float bias = d_bias[index];
    d_bias[index] = (0.2 * bias) + 0.4 * (d_diff[index] + bias);
  }

}

__global__ void decrement_bias(float * d_bias, float * d_diff, unsigned int ndim)
{
  unsigned int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if( index < ndim ){
    float bias = d_bias[index];
    d_bias[index] = bias - 0.4 * (d_diff[index] + bias);
  }
}

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
){
  unsigned int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float solution;

  if( index < n_dim )
  {
		if( direction == 0 )
    {
			curandState l_state;
			l_state = g_state[index];
      d_diff[index] = curand_normal(&l_state) * delta;
      solution = d_sol[index] + d_bias[index] + d_diff[index];
			g_state[index] = l_state;
		}
    else
    {
			solution = d_sol[index] - d_bias[index] - d_diff[index];
		}
    // Check bounds of the new solution
		solution = max(x_min, solution);
		solution = min(x_max, solution);


    d_new_solution[index] = solution;
	}
}
