#ifndef GLOBALS_CUH
#define GLOBALS_CUH
#include <cuda_runtime.h> 
#include <cuda.h>


int CUDA_SAFE_CALL(CUresult err);
int CUDA_SAFE_CALL(cudaError_t err);

template<typename T, typename R, R (*CleanupFunction)(T)>
class Collection {
private:
  std::map<uint32_t, T> data;
  uint32_t counter = 0;

public:
  uint32_t add(const T element, uint32_t key = 0) {
    if(key){
      if(data.find(key) != data.end()){
        std::cerr << "Key of "<< key <<" Already exists"<<std::endl;
        return 0;
      }
      data[key] = element;
      return key;
    } else {
      while (data.find(++counter) != data.end());
      data[counter] = element;
      return counter;
    }
  }
  T get(uint32_t key){
    auto pair = data.find(key);
    if(pair != data.end())
      return data.find(key)->second;
    std::cerr << "Cannot find element corresponding to "<<key;
    return 0;
  }
  T destroy(uint32_t key, bool handled = false) {
    auto it = data.find(key);
    if (it != data.end()) {
      if(!handled) CleanupFunction(it->second);
      data.erase(key);
    } else {
      std::cerr<<"bad key";
    }
    return it->second;
  }
  ~Collection() {
    for (auto& pair : data) {
      CleanupFunction(pair.second);
    }
    data.clear();
  }
};

namespace Globals {
  extern Collection<void*, cudaError_t, &cudaFree> devptrs;
  extern Collection<CUmodule, CUresult, &cuModuleUnload> modules;
  void donothing(CUfunction);
  extern Collection<CUfunction, void, donothing> functions;
}

#endif