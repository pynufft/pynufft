R="""
KERNEL void cSqrt( 
         GLOBAL_MEM  ${ctype} *CX)
{
// Copy x to y: y = x;
//CX: input output array (${ctype})

int gid=get_global_id(0);  
CX[gid].x=sqrt(CX[gid].x);
};
"""  
scalar_arg_dtypes=[None,]