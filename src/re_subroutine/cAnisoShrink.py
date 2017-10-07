"""
cAnisoShrink
======================================
KERNEL void cAnisoShrink(const  float2 threshold,
                                GLOBAL_MEM const float2 *indata,
                                GLOBAL_MEM  float2 *outdata)
"""


from numpy import complex64
R="""
KERNEL void cAnisoShrink(const  float2 threshold,
                                GLOBAL_MEM const float2 *indata,
                                GLOBAL_MEM  float2 *outdata)
{
const uint gid =  get_global_id(0); 
float2 tmp; // temporay register
tmp = indata[gid];
//float zero = 0.0;
//tmp.x=sign(tmp.x)*max(fabs(tmp.x)-threshold.x, zero); 
//tmp.y=sign(tmp.y)*max(fabs(tmp.y)-threshold.y, zero); 
tmp.x =  (tmp.x > threshold.x)*(tmp.x - threshold.x) ;//+ (tmp.x < - threshold.x)*(tmp.x + threshold.x);
tmp.y =  (tmp.y > threshold.x)*(tmp.y - threshold.x) ;//+ (tmp.y < - threshold.x)*(tmp.y + threshold.x);
outdata[gid]=tmp;
};
"""
scalar_arg_dtypes=[complex64, None, None]