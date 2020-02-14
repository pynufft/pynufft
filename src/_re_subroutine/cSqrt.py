R="""
KERNEL void cSqrt( 
         GLOBAL_MEM  float2 *CX)
{
// Copy x to y: y = x;
//CX: input output array (float2)

int gid=get_global_id(0);  
CX[gid].x=sqrt(CX[gid].x);
};
"""  
scalar_arg_dtypes=[None,]