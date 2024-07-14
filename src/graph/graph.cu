#include <vector>
#include <iostream>

#include "../base/globals.cuh"

namespace GraphThings{
  //Serialized graph formatted like so:
  //Initcb number - 8
  //Numnodes - 4
  //Reserved - 4
  //Nodes:
    //type - 4
    //ndeps - 4
    //deps - 4*ndeps (Integer index of dep)
    //Info (switch):
      //Alloc: (12)
        //ptr ID - 4
        //Bytesize - 8
      //Free: (4)
        //ptr ID - 4
      //Kernel (24+variable)
        //func ID - 4
        //numargs - 4
        //Blockdim (4,2,2)
        //Griddim (4,2,2)
        //args: 
          //Argsize: 2
          //Arg: argsize
      //Load (8+variable)
        //ptr - 4
        //size - 4
        //mem - size

  cudaGraphNode_t makeEmpty(cudaGraph_t graph, std::vector<cudaGraphNode_t> deps, char** mem){
    cudaGraphNode_t node;
    CUDA_SAFE_CALL(cudaGraphAddEmptyNode(&node, graph, deps.data(), deps.size()));
    return node;
  }
  cudaGraphNode_t makeAlloc(cudaGraph_t graph, std::vector<cudaGraphNode_t> deps, char** mem){
    cudaMemAllocNodeParams allocParams;
    memset(&allocParams, 0, sizeof(allocParams));
    allocParams.bytesize = *(uint64_t*) (*mem+4);
    allocParams.poolProps.allocType = cudaMemAllocationTypePinned;
    allocParams.poolProps.location.id = 0;
    allocParams.poolProps.location.type = cudaMemLocationTypeDevice;

    cudaGraphNode_t node;
    CUDA_SAFE_CALL(cudaGraphAddMemAllocNode(&node, graph, deps.data(), deps.size(), &allocParams));
    Globals::devptrs.add(allocParams.dptr, *(uint32_t*)*mem);
    *mem += 12;
    return node;
  }
  cudaGraphNode_t makeFree(cudaGraph_t graph, std::vector<cudaGraphNode_t> deps, char** mem){
    cudaGraphNode_t node;
    CUDA_SAFE_CALL(cudaGraphAddMemFreeNode(
      &node, graph, deps.data(), deps.size(), 
      Globals::devptrs.destroy(*(uint32_t*)*mem, true)
    ));
    *mem += 4;
    return node;
  }

  cudaGraphNode_t makeCall(cudaGraph_t graph, std::vector<cudaGraphNode_t> deps, char** mem){
    uint32_t funcid = *(int*)*mem;
    uint32_t argnum = *(int*)(*mem+4);
    void** ptrs = (void**)malloc(sizeof(void*)*argnum); //not always used
    void** args = (void**)malloc(sizeof(void*)*argnum);
    *mem += 8;

    cudaKernelNodeParams callParams = {0};
    callParams.func = Globals::functions.get(funcid);
    callParams.blockDim = dim3(*(uint32_t*)*mem, *(uint16_t*)(*mem+4), *(uint16_t*)(*mem+6));
    *mem += 8;
    callParams.gridDim = dim3(*(uint32_t*)*mem, *(uint16_t*)(*mem+4), *(uint16_t*)(*mem+6));
    *mem += 8;
    callParams.sharedMemBytes = 0;
    callParams.extra = NULL;

    for(int i=0; i<argnum; i++){
      uint16_t size = *(uint16_t*)*mem;
      *mem+=2;
      if(size == 0){
        ptrs[i] = Globals::devptrs.get(*(uint32_t*)*mem);
        args[i] = &ptrs[i];
        *mem += 8;
      } else {
        args[i] = *mem;
        *mem += size;
      }
    }
    
    callParams.kernelParams = args;
    cudaGraphNode_t node;
    CUDA_SAFE_CALL(cudaGraphAddKernelNode(&node, graph, deps.data(), deps.size(), &callParams));

    free(args);
    free(ptrs);
    return node;
  }
  void freeBuf(char* buf){
    std::cout <<"freeing"<< buf << std::endl;
    free(buf);
  }
  cudaGraphNode_t makeLoadDir(cudaGraph_t graph, std::vector<cudaGraphNode_t> deps, char** mem){
    uint32_t size = *(uint32_t*)*mem;
    char* buf = (char*)malloc(size);
    std::cout <<"allocating"<< buf << std::endl;
    memcpy(buf, *mem+8, size);
    cudaGraphNode_t node;
    CUDA_SAFE_CALL(cudaGraphAddMemcpyNode1D(
      &node, graph, deps.data(), deps.size(),
      Globals::devptrs.get(*(uint32_t*)*mem), buf, size, cudaMemcpyHostToDevice
    ));

    cudaGraphNode_t freeNode;
    cudaHostNodeParams freeParams = {0};
    freeParams.fn = &free;
    freeParams.userData = &buf;
    CUDA_SAFE_CALL(cudaGraphAddHostNode(&freeNode, graph, &node, 1, &freeParams));

    *mem += 8+size;
    return node;
  }

  cudaGraphNode_t (*graphFuncs[])(cudaGraph_t, std::vector<cudaGraphNode_t>, char**) = {
    makeEmpty,
    makeAlloc,
    makeFree,
    makeCall,
    makeLoadDir,
  };
  void parseGraph(char* gdesc){
    char* mem = mem+16;
    std::vector<cudaGraphNode_t> nodes;
    std::vector<cudaGraphNode_t> deps;
    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);

    for(int i=0; i<*(uint32_t*)(gdesc+8); i++){
      int type = *(int*)mem;
      int ndeps = *(int*)(mem+4);
      mem += 8;
      for(int j=0; j<ndeps; j++){
        deps.push_back(nodes[*(uint32_t*)mem]);
        mem+=4;
      }
      nodes.push_back(graphFuncs[type](graph, deps, &mem));
      deps.clear();
    }
    cudaGraphExec_t graphExec;
    CUDA_SAFE_CALL(cudaGraphInstantiate(&graphExec, graph));
    CUDA_SAFE_CALL(cudaGraphLaunch(graphExec, NULL));

    free(gdesc);
  }
}