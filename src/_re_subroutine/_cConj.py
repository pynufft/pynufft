R="""
KERNEL void cConj( 
         GLOBAL_MEM  float2 *CX)
{
// return the conjugate of x: x = conj(x);
//CX: input(output) array (float2)
int gid=get_global_id(0);  
CX[gid].y =   - CX[gid].y;
};
"""  
scalar_arg_dtypes=[None, None]