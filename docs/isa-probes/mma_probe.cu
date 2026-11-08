template<int KIND> struct Probe;
template<> struct Probe<0> { // mxf8f6f4 fp8 block-scale
  __device__ static void run(float&c0,float&c1,float&c2,float&c3,unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1,unsigned s){
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};" : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(s),"h"(ush()),"h"(ush()),"r"(s),"h"(ush()),"h"(ush()));
  }};
template<> struct Probe<1> { // mxf4nvf4 nvfp4 block-scale
  __device__ static void run(float&c0,float&c1,float&c2,float&c3,unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1,unsigned s){
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};" : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(s),"h"(ush()),"h"(ush()),"r"(s),"h"(ush()),"h"(ush()));
  }};
template<> struct Probe<2> { // plain fp8
  __device__ static void run(float&c0,float&c1,float&c2,float&c3,unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1,unsigned s){
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};" : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
  }};
