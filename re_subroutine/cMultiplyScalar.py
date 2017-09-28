R="""
KERNEL void cMultiplyScalar(
             const ${ctype} CA,
        GLOBAL_MEM ${ctype} *CX)
{ 
// Scale CX by CA: CX=CA*CX
//  CA: scaling factor(${ctype})
//*CX: input, output array(${ctype})
int gid = get_global_id(0);  
CX[gid].x=CA.x*CX[gid].x-CA.y*CX[gid].y;
CX[gid].y=CA.x*CX[gid].y+CA.y*CX[gid].x;
};           
"""
from numpy import complex64  
scalar_arg_dtypes=[complex64, None]