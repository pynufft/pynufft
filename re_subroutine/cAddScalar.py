from numpy import complex64
R="""
KERNEL void cAddScalar(const ${ctype} CA,
                                    GLOBAL_MEM ${ctype} *CX)
{ 
// (single complex) scale x by a: x = x + ca;
// CA: add factor 
// CX: input and output array (${ctype})
int gid = get_global_id(0);  
CX[gid].x += CA.x;
CX[gid].y += CA.y;
};
"""
scalar_arg_dtypes=[complex64, None]
