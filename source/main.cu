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

  SolisWets *sw = new SolisWets(5, 3, 50, 32, 2, -100.0, +100.0, 0.4);

  int n_dim = 50;
  thrust::device_vector<float> sol(n_dim);
  thrust::counting_iterator<unsigned int> index_sequence_begin(0);

  thrust::transform(index_sequence_begin, index_sequence_begin + n_dim, sol.begin(), prg(-100.0f, +100.0f));

  sw->optimize(20000, thrust::raw_pointer_cast(sol.data()));

  return 0;
}
