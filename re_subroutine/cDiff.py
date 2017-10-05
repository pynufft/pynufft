R="""
 KERNEL void cDiff(  GLOBAL_MEM    const    int            *order2,
                                        GLOBAL_MEM     const   float2     *indata,
                                        GLOBAL_MEM                  float2     *outdata)
{
const uint gid =  get_global_id(0); 
const uint ind = order2[gid];
outdata[gid]=indata[ind]- indata[gid];
};
"""
scalar_arg_dtypes=[None, None, None]