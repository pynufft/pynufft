R="""
KERNEL void cMultiplyVec( GLOBAL_MEM ${ctype} *a,
                                    GLOBAL_MEM ${ctype} *b,
                                    GLOBAL_MEM ${ctype} *dest)
{    const int i = get_global_id(0);
    dest[i].x = a[i].x*b[i].x-a[i].y*b[i].y;
    dest[i].y = a[i].x*b[i].y+a[i].y*b[i].x;
    //barrier(CLK_GLOBAL_MEM_FENCE); 
};
"""
scalar_arg_dtypes=[None, None, None]