R="""
KERNEL void cMultiplyScalar(
             const float2 CA,
        GLOBAL_MEM float2 *CX)
{ 
// Scale CX by CA: CX=CA*CX
//  CA: scaling factor(float2)
//*CX: input, output array(float2)
int gid = get_global_id(0);  
CX[gid].x=CA.x*CX[gid].x-CA.y*CX[gid].y;
CX[gid].y=CA.x*CX[gid].y+CA.y*CX[gid].x;
};           
"""
from numpy import complex64  
scalar_arg_dtypes=[complex64, None]