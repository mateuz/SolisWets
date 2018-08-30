#include "soliswets.cuh"
#include <random>
#include <cassert>

SolisWets::SolisWets(
  unsigned int _n_success,
  unsigned int _n_fails,
  unsigned int _n_dim,
  unsigned int _n_threads,
  unsigned int _n_blocks,
  unsigned int _f_id,
  float _x_min,
  float _x_max,
  float _delta
){
  n_success = _n_success;
  n_fails   = _n_fails;
  n_dim     = _n_dim;
  n_threads = _n_threads;
  n_blocks  = _n_blocks;
  f_id      = _f_id;
  x_min     = _x_min;
  x_max     = _x_max;
  delta     = _delta;

  checkCudaErrors(cudaMalloc((void **)&d_new_solution, n_dim * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_bias, n_dim * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_diff, n_dim * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_states, n_dim * sizeof(curandStateXORWOW_t)));

  Configuration conf;
  conf.x_min = x_min;
  conf.x_max = x_max;
  conf.ps = 1;
  conf.n_dim = n_dim;
  checkCudaErrors(cudaMemcpyToSymbol(params, &conf, sizeof(Configuration)));

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

__global__ void s_rosenbrock_kernel(
  float * x,
  float * fitness,
  uint ndim
){
 uint id_p = threadIdx.x + (blockIdx.x * blockDim.x);
 uint id_d = id_p * ndim;

 float s1 = 0.0, s2 = 0.0, a, b;

 uint i = 0;
 for(; i < (ndim - 1); i++){
   a = x[id_d + i] + 1.0f;
   b = x[id_d + i + 1] + 1.0f;

   s1  = ( a * a ) - b;
   s2 += ( 100.0 * s1 * s1 );

   s1  = (a - 1.0);
   s2 += ( s1 * s1 );
 }

 *fitness = s2;
}

__global__ void s_sphera_kernel(
  float * x,
  float * fitness,
  uint ndim
){
 uint id_p = threadIdx.x + (blockIdx.x * blockDim.x);
 uint id_d = id_p * ndim;

 float s = 0.0, a;

 uint i = 0;
 for(; i < (ndim - 1); i++){
   a = x[id_d + i];
   s += a * a;
 }

 *fitness = s;
}

float SolisWets::optimize(const unsigned int n_evals, float * d_sol){
  Benchmarks * B = new F2(n_dim, 1);

  unsigned int _n_success = 0, _n_fails = 0, counter = 0;

  float * d_fitness;
  checkCudaErrors(cudaMalloc((void **)&d_fitness, sizeof(float)));

  float result = 0.0, current_fitness = 0.0;

  //eval solution
  B->compute(d_sol, d_fitness);
  //s_rosenbrock_kernel<<<1, 1>>>(d_sol, d_fitness, n_dim);
  //checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMemcpy(&current_fitness, d_fitness, sizeof(float), cudaMemcpyDeviceToHost));
  printf("%.2lf\n", current_fitness);

  for( unsigned int it = 1; it <= n_evals; ){
    generate_new_solution<<<n_blocks, n_threads>>>(
      d_sol, d_new_solution, d_bias,
      d_diff, delta, 0, d_states);

    checkCudaErrors(cudaGetLastError());

    B->compute(d_new_solution, d_fitness);

    checkCudaErrors(cudaMemcpy(&result, d_fitness, sizeof(float), cudaMemcpyDeviceToHost));

    it++;

    if( result < current_fitness ){
      printf("[+] %-3.2f < %-3.2f | [%-5u]\n", result, current_fitness, it);
      current_fitness = result;

      //increment bias and copy
      increment_bias<<<n_blocks, n_threads>>>(d_bias, d_diff, n_dim);
      checkCudaErrors(cudaGetLastError());

      checkCudaErrors(cudaMemcpy(d_sol, d_new_solution, n_dim * sizeof(float), cudaMemcpyDeviceToDevice));

      _n_success++;
      _n_fails = 0;
      counter = 0;
    } else {
      if( it >= n_evals ) break;

      generate_new_solution<<<n_blocks, n_threads>>>(
        d_sol, d_new_solution, d_bias,
        d_diff, delta, 1, d_states);

      checkCudaErrors(cudaGetLastError());

      B->compute(d_new_solution, d_fitness);

      checkCudaErrors(cudaMemcpy(&result, d_fitness, sizeof(float), cudaMemcpyDeviceToHost));
      it++;

      if( result <= current_fitness ){
        printf("[-] %-3.2f < %-3.2f | [%-5u]\n", result, current_fitness, it);

        current_fitness = result;

        //decrement bias and copy
        decrement_bias<<<n_blocks, n_threads>>>(d_bias, d_diff, n_dim);
        checkCudaErrors(cudaGetLastError());

        checkCudaErrors(cudaMemcpy(d_sol, d_new_solution, n_dim * sizeof(float), cudaMemcpyDeviceToDevice));

        _n_success++;
        _n_fails = 0;
        counter = 0;
      } else {
        _n_success = 0;
        _n_fails++;

        counter++;
      }
    }
    if( _n_success > n_success ){
      delta *= 2.0;
      _n_success = 0;
    } else if( _n_fails > n_fails ){
      delta /= 2;
      _n_fails = 0;
    }

    if( counter == 100 ){
      counter = 0;
      delta = 0.2 * (x_max - x_min);
      checkCudaErrors(cudaMemset(d_bias, 0, n_dim * sizeof(float)));
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
  float delta,
  unsigned short direction,
  curandState * g_state
){
  unsigned int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float solution;

  if( index < params.n_dim )
  {
    if( direction == 0 )
    {
      curandState l_state;
      l_state = g_state[index];
      d_diff[index] = curand_normal(&l_state) * delta;
      solution = d_sol[index] + d_bias[index] + d_diff[index];
      g_state[index] = l_state;
    } else {
      solution = d_sol[index] - d_bias[index] - d_diff[index];
    }

    // Check bounds of the new solution
    solution = max(params.x_min, solution);
    solution = min(params.x_max, solution);
    d_new_solution[index] = solution;
  }
}
