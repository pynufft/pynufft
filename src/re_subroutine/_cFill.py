R="""
KERNEL void cFill( 
             const float2 CA,
        GLOBAL_MEM float2 *CX)
{
// Fill CX with CA: x = conj(x);
//CX: input(output) array (float2)
int gid=get_global_id(0);  
CX[gid] =   CA;
};
"""  
scalar_arg_dtypes=[None, None]