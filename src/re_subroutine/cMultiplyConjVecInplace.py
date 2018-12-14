R="""
KERNEL void cMultiplyConjVecInplace( 
        GLOBAL_MEM float2 *a,
        GLOBAL_MEM float2 *outb)
{
const uint gid = get_global_id(0);
float2 mul = a[gid]; //  taking the conjugate
float2 orig = outb[gid];
float2 tmp;
tmp.x = orig.x * mul.x + orig.y * mul.y;
tmp.y = - orig.x * mul.y + orig.y * mul.x; 
outb[gid] = tmp;
};
"""
scalar_arg_dtypes=[None, None, None]