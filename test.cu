__global__
void add(float* A, float* B, float* C, size_t N){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    for(size_t i=idx; i<N; i+=blockDim.x*gridDim.x){
        C[i]=A[i]+B[i];
    }
}
__global__
void fill(float* A, float val, size_t N){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    for(size_t i=idx; i<N; i+=blockDim.x*gridDim.x){
        A[i]=val;
    }
}