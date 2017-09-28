R="""
KERNEL void cCopy( 
         GLOBAL_MEM  const ${ctype} *CX,
         GLOBAL_MEM             ${ctype} *CY)
{
// Copy x to y: y = x;
//CX: input array (${ctype})
// CY output array (${ctype})
int gid=get_global_id(0);  
CY[gid]=CX[gid];
};
"""  
scalar_arg_dtypes=[None, None]