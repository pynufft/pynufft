"""
cAddScalar
=================================
KERNEL void cAddScalar(const float2 CA,   
                    GLOBAL_MEM float2 *CX)
Offload add to heterogeneous devices.
"""

from numpy import complex64
R="""
KERNEL void cAddScalar(const float2 CA,
                                    GLOBAL_MEM float2 *CX)
{ 
// (single complex) scale x by a: x = x + ca;
// CA: add factor 
// CX: input and output array (float2)
int gid = get_global_id(0);  
CX[gid].x += CA.x;
CX[gid].y += CA.y;
};
"""
scalar_arg_dtypes=[complex64, None]
