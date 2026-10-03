#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "cuda_kernel.cuh"
#define CK(x) do { if ((x)!=cudaSuccess) { std::fprintf(stderr,"CUDA failed at %d\n",__LINE__); return 2; } } while(0)
#define BL(x) do { if ((x)!=CUBLAS_STATUS_SUCCESS) return 3; } while(0)
int main() {
    const int M=1024,N=1024,K=1024;
    std::vector<float> a(M*K),b(K*N),c(M*N),out(M*N),ref(M*N);
    for (int i=0;i<M*K;++i) a[i]=float(i%31-15)/16;
    for (int i=0;i<K*N;++i) b[i]=float(i%29-14)/16;
    for (int i=0;i<M*N;++i) c[i]=float(i%13-6)/8;
    float *da,*db,*dc,*dr;
    CK(cudaMalloc(&da,a.size()*4)); CK(cudaMalloc(&db,b.size()*4));
    CK(cudaMalloc(&dc,c.size()*4)); CK(cudaMalloc(&dr,c.size()*4));
    CK(cudaMemcpy(da,a.data(),a.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db,b.data(),b.size()*4,cudaMemcpyHostToDevice));
    cublasHandle_t handle; BL(cublasCreate(&handle)); BL(cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH));
    for (float beta : {0.0f,1.0f,-0.5f}) {
        float alpha=0.75f;
        CK(cudaMemcpy(dc,c.data(),c.size()*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dr,c.data(),c.size()*4,cudaMemcpyHostToDevice));
        cuda_gemm(M,N,K,alpha,da,db,beta,dc);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        BL(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,db,N,da,K,&beta,dr,N));
        CK(cudaMemcpy(out.data(),dc,c.size()*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(ref.data(),dr,c.size()*4,cudaMemcpyDeviceToHost));
        double max_error=0;
        for (int i=0;i<M*N;++i) {
            double err=std::abs(double(out[i])-ref[i]);
            if(err>max_error) max_error=err;
            if(!std::isfinite(out[i]) || err>1e-4+1e-4*std::abs(ref[i])) {
                std::printf("FAIL beta=%g at=%d actual=%g ref=%g\n",beta,i,out[i],ref[i]); return 1;
            }
        }
        std::printf("PASS alpha=%g beta=%g max_abs_error=%g\n",alpha,beta,max_error);
    }
    BL(cublasDestroy(handle)); CK(cudaFree(da)); CK(cudaFree(db)); CK(cudaFree(dc)); CK(cudaFree(dr));
}
