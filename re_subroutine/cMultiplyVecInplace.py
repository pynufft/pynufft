R="""
KERNEL void cMultiplyVecInplace( 
        GLOBAL_MEM ${ctype} *a,
        GLOBAL_MEM ${ctype} *outb)
{
const int gid = get_global_id(0);
${ctype} mul=a[gid];
${ctype} orig = outb[gid];
${ctype} tmp;
tmp.x=orig.x*mul.x-orig.y*mul.y;
tmp.y=orig.x*mul.y+orig.y*mul.x; 
outb[gid]=tmp;
};
"""
scalar_arg_dtypes=[None, None, None]