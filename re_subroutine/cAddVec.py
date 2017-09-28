R="""
KERNEL void cAddVec( 
        GLOBAL_MEM ${ctype} *a,
        GLOBAL_MEM ${ctype} *b,
        GLOBAL_MEM ${ctype} *dest)
{const int i = get_global_id(0);
dest[i]= a[i]+b[i];
};
"""
scalar_arg_dtypes=[None, None, None]