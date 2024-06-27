#include <iostream>
#include <map>
#include <cuda_runtime.h> 
#include <cuda.h>

#include "globals.cuh"

int CUDA_SAFE_CALL(CUresult err){
  if (cudaSuccess != err) {                                         
    const char *errStr;
    cuGetErrorString(err, &errStr);
    fprintf(stderr, "CUDA Error: %s (error code: %d)\n", errStr, err);
  } 
  return err;
} 
int CUDA_SAFE_CALL(cudaError_t err){
  if(CUDA_SUCCESS != err) {
    fprintf(stderr, "CUDA Error_t:  %d\n", err);
  }
  return err;
}


namespace Globals {
  void donothing(CUfunction func) {
      // Who woulda guessed
  }

  // Definitions of global objects
  Collection<void*, cudaError_t, &cudaFree> devptrs;
  Collection<CUmodule, CUresult, &cuModuleUnload> modules;
  Collection<CUfunction, void, donothing> functions;
}