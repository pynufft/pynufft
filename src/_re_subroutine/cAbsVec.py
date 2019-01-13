"""
KERNEL void cAbsVec( GLOBAL_MEM const float2 *indata, 
                                GLOBAL_MEM  float2 *outdata)
Offload abs to heterogeneous devices.
"""
R="""
KERNEL void cAbsVec( GLOBAL_MEM const float2 *indata, 
                                    GLOBAL_MEM            float2 *outdata)
{
    const int gid =  get_global_id(0);
    float2 tmp = indata[gid];
    tmp.x = sqrt( tmp.x*tmp.x+tmp.y*tmp.y);
    //tmp.x =  sqrt(tmp.x);
    tmp.y = 0.0;
    outdata[gid]=tmp;
};
"""
scalar_arg_dtypes=[None, None]