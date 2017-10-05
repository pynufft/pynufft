R="""
KERNEL void cCopy( 
         GLOBAL_MEM  const float2 *CX,
         GLOBAL_MEM             float2 *CY)
{
// Copy x to y: y = x;
//CX: input array (float2)
// CY output array (float2)
int gid=get_global_id(0);  
CY[gid]=CX[gid];
};
"""  
scalar_arg_dtypes=[None, None]