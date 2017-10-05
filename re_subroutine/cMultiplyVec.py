R="""
KERNEL void cMultiplyVec( GLOBAL_MEM float2 *a,
                                    GLOBAL_MEM float2 *b,
                                    GLOBAL_MEM float2 *dest)
{    const int i = get_global_id(0);
    dest[i].x = a[i].x*b[i].x-a[i].y*b[i].y;
    dest[i].y = a[i].x*b[i].y+a[i].y*b[i].x;
    //barrier(CLK_GLOBAL_MEM_FENCE); 
};
"""
scalar_arg_dtypes=[None, None, None]