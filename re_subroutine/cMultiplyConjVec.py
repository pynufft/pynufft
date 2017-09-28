R="""
KERNEL void cMultiplyConjVec( 
        GLOBAL_MEM ${ctype} *a,
        GLOBAL_MEM ${ctype} *b,
        GLOBAL_MEM ${ctype} *dest)
{// dest[i]=conj(a[i]) * b[i] 
const int i=get_global_id(0);
dest[i].x=a[i].x*b[i].x+a[i].y*b[i].y;
dest[i].y=a[i].x*b[i].y-a[i].y*b[i].x;
};
"""
scalar_arg_dtypes=[None, None, None]