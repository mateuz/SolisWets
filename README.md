# GPU Solis Wets Optimization Algorithm

###### Simple implementation of Solis-Wets optimization algorithm accordly Minimization by Random Techniques [1] on GPU using CUDA [2]. The parallel idea in this case is optimize each dimension of the optimization problem by a CUDA thread. Simple, easy and fast.

***

##### Compile

```sh
$ cd repo
$ make
```
##### Execute

```sh
$ cd repo
$ ./sw-optimizer or make run
```

#### Clean up

```sh
$ make clean
```

### TODO

    - Empty list for a while
    
***

[1] [Minimization by Random Search Techniques by Francisco J. Solis and Roger J-B. Wets (1981)](https://www.math.ucdavis.edu/~rjbw/mypage/Miscellaneous_files/randSearch.pdf)

[2] [CUDA is a parallel computing platform and programming model developed by NVIDIA for GPGPU.](https://developer.nvidia.com/cuda-zone)
