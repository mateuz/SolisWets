#include <stdio.h>
#include "soliswets.cuh"

#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <iostream>

struct prg
{
  float a, b;

  __host__ __device__ prg(float _a=0.f, float _b=1.f) : a(_a), b(_b) {};

  __host__ __device__ float operator()(const unsigned int n) const
  {
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> dist(a, b);
    rng.discard(n);

    return dist(rng);
  }
};

int main(int argc, char * argv[]){

  uint n_dim     = 1000;
  uint n_success = 5;
  uint n_fails   = 3;
  uint n_threads = 32;
  uint n_blocks  = 32;

  float x_min = -100.0;
  float x_max = +100.0;
  float delta = 0.4 * (x_max - x_min);

  SolisWets *sw = new SolisWets(n_success, n_fails, n_dim, n_threads, n_blocks, x_min, x_max, delta);

  thrust::device_vector<float> sol(n_dim);
  thrust::counting_iterator<unsigned int> index_sequence_begin(0);

  thrust::transform(index_sequence_begin, index_sequence_begin + n_dim, sol.begin(), prg(x_min, x_max));

  sw->optimize(100000, thrust::raw_pointer_cast(sol.data()));

  return 0;
}
