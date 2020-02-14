"""
KERNEL void cAddVec( 
        GLOBAL_MEM float2 *a,
        GLOBAL_MEM float2 *b,
        GLOBAL_MEM float2 *dest)
        Add two vectors
"""

R="""
KERNEL void cAddVec( 
        GLOBAL_MEM float2 *a,
        GLOBAL_MEM float2 *b,
        GLOBAL_MEM float2 *dest)
{const int i = get_global_id(0);
dest[i]= a[i]+b[i];
};
"""
scalar_arg_dtypes=[None, None, None]