#include <cuda_bf16.h>
__global__ void t(float* o, unsigned packed){
  unsigned r;
  asm("cvt.rn.bf16x2.e2m1x2 %0, %1;" : "=r"(r) : "r"(packed));
  o[threadIdx.x] = __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&r));
}
