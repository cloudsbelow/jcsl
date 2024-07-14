#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda.h>
#include <map>
#include <fcntl.h>
#include <io.h>

#include "src/base/globals.cuh"

struct header{
  uint32_t protocol;
  uint32_t info;
  uint64_t size;
};
enum Command{
  DevMalloc,
  DevFree,
  CreateModule,
  DestroyModule,
  CreateFunc,
  CallFunc,
  GetBuf,
  LoadBuf,
};
//there is no way we would get f***ed by struct padding, right?
struct command{
  uint64_t commandID; //This is unique across commands
  Command command;
  uint32_t arg1;
  uint32_t arg2;
  uint32_t arg3;
};
struct response{
  uint64_t commandID;
  uint32_t err;
  uint32_t res;
};



void createModule(response &r, command c, char* mem){
  //doing this up here stops compiler warnings?
  char* ptxcode = (char*) malloc(c.arg1+1);
  memcpy(ptxcode,mem,c.arg1);
  ptxcode[c.arg1] = 0;

  CUmodule module;
  r.err = CUDA_SAFE_CALL(cuModuleLoadData(&module, ptxcode));
  r.res = Globals::modules.add(module);
  free(ptxcode);
}
void createFunction(response &r, command c, char*mem){
  char* funcname = (char*) malloc(c.arg2+1);
  memcpy(funcname, mem, c.arg2);
  funcname[c.arg2] = 0;

  CUfunction kernelFunc;
  r.err = CUDA_SAFE_CALL(cuModuleGetFunction(&kernelFunc, Globals::modules.get(c.arg1), funcname));
  r.res = Globals::functions.add(kernelFunc, c.arg3);
  free(funcname);
}
void callFunction(response &r, command c, char* mem){
  void** ptrs = (void**)malloc(sizeof(void*)*c.arg2); //not always used
  void** args = (void**)malloc(sizeof(void*)*c.arg2);
  int offset = 0;
  for(int i=0; i<c.arg2; i++){
    uint16_t size = *(uint16_t*)(mem+offset);
    offset+=2;
    if(size == 0){
      ptrs[i] = Globals::devptrs.get(*(uint32_t*)(mem+offset));
      args[i] = &ptrs[i];
      offset += 8;
    } else {
      args[i] = mem+offset;
      offset+=size;
    }
  }
  r.err = CUDA_SAFE_CALL(cuLaunchKernel(Globals::functions.get(c.arg1), 10,1,1, 32,1,1, 0, NULL, args, NULL));
  //cudaDeviceSynchronize();
  free(args);
  free(ptrs);
}

void createDevptr(response &r, command c){
  void* devptr;
  if((r.err=cudaMalloc(&devptr, *(uint64_t*)&c.arg2))!=CUDA_SUCCESS){
    std::cerr<<"bad malloc";
  }
  r.res = Globals::devptrs.add(devptr, c.arg1);
}
int printBuffer(response &r, command c, char** resmem){
  *resmem = (char*)malloc(c.arg2);
  cudaDeviceSynchronize();
  if((r.err=cudaMemcpy(*resmem, (char*)Globals::devptrs.get(c.arg1)+c.arg3, c.arg2, cudaMemcpyDeviceToHost))!=CUDA_SUCCESS){
    std::cerr<<"bad memcpy - could not download buffer";
    free(*resmem);
    return 0;
  } else {
    r.res = c.arg2;
    return c.arg2;
  }
}
void loadBuffer(response &r, command c, char* mem){
  if((r.err = cudaMemcpy((char*)Globals::devptrs.get(c.arg1)+c.arg3, mem, c.arg2, cudaMemcpyHostToDevice))!=CUDA_SUCCESS){
    std::cerr<<"bad memcpy - could not upload buffer";
  }
}

void handleCommand(command c, char* mem){
  response r;
  r.commandID = c.commandID;
  r.err = 0;
  char* resmem;
  int resmemlen=0;
  switch(c.command){
    case DevMalloc:
      createDevptr(r, c);
      break;
    case DevFree:
      Globals::devptrs.destroy(c.arg1);
      break;
    case CreateModule:
      createModule(r, c, mem);
      break;
    case DestroyModule:
      Globals::modules.destroy(c.arg1);
      break;
    case CreateFunc:
      createFunction(r,c,mem);
      break;
    case CallFunc:
      callFunction(r,c,mem);
      break;
    case GetBuf:
      resmemlen = printBuffer(r,c, &resmem);
      break;
    case LoadBuf:
      loadBuffer(r, c, mem);
      break;

    default:
      std::cerr<<"no such command "<<c.command;
  }

  char buffer[sizeof(response)];
  std::memcpy(buffer, &r, sizeof(response));
  std::fwrite(buffer, sizeof(response), 1, stdout);
  if(resmemlen>0){
    std::fwrite(resmem, resmemlen, 1, stdout);
    free(resmem);
  }
  std::fflush(stdout);
}

int main() {
  _setmode(_fileno(stdin), _O_BINARY);
  _setmode(_fileno(stdout), _O_BINARY);
  float* A;
  cudaMalloc(&A, 4*64); //we need to have this, otherwise the program errors. Don't ask.
  
  while(!std::cin.eof()){
    header h;
    std::cin.read((char*)&h, sizeof(h));
    void* data = malloc(h.size);
    std::cin.read((char*) data, h.size);
    switch (h.info){
      case 0: //echo
        std::cerr<<(uint8_t*)data;
        break;
      case 1: //execute one command
        handleCommand(*(command*)data, (char*)data+sizeof(command));
        break;
      case 8: //Kill the program
        free(data); cudaFree(A);
        exit(0);
        break;
      default:
        std::cerr<<"unknown header instruction "<<h.info;
    }
    free(data);
  }
  cudaFree(A);
  return 0;
}