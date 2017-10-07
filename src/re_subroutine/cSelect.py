"""
KERNEL void cSelect(
    GLOBAL_MEM const  int *order1,
    GLOBAL_MEM const  int *order2,
    GLOBAL_MEM const float2 *indata,
    GLOBAL_MEM       float2 *outdata)
"""


R="""
KERNEL void cSelect(
    GLOBAL_MEM const  int *order1,
    GLOBAL_MEM const  int *order2,
    GLOBAL_MEM const float2 *indata,
    GLOBAL_MEM       float2 *outdata)
{
const uint gid=get_global_id(0); 
outdata[order2[gid]]=
               indata[order1[gid]];
};
"""
scalar_arg_dtypes=[None, None, None, None]