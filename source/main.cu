#include <stdio.h>
#include "soliswets.cuh"

int main(int argc, char * argv[]){

  SolisWets *sw = new SolisWets(5, 3, 100, 64, 2, -100.0, +100.0, 0.4);

  return 0;
}
