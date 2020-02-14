"""
cHypot
======================================
KERNEL void cHypot(GLOBAL_MEM float2 *x,
                                GLOBAL_MEM const float2 *y)
"""
from numpy import complex64
R="""
KERNEL void cHypot(GLOBAL_MEM float2 *x,
                                GLOBAL_MEM const float2 *y)
{
const unsigned int gid =  get_global_id(0); 
float2 tmp_x;
float2 tmp_y;
tmp_x = x[gid];
tmp_y = y[gid];
tmp_x.x = hypot( tmp_x.x, tmp_x.y); // sqrt( tmp_x.x*tmp_x.x + tmp_x.y*tmp_x.y);
tmp_y.x = hypot( tmp_y.x, tmp_y.y); // sqrt( tmp_y.x*tmp_y.x + tmp_y.y*tmp_y.y);


//float z;
//float r;

//z = (tmp_x.x > tmp_y.x) * tmp_x.x + (tmp_x.x < tmp_y.x) * tmp_y.x;

//r = tmp_x.x + tmp_y.y - z;
 
//r /= z;

//x[gid].x = z*sqrt(1.0 + r*r);

x[gid].x = hypot(tmp_x.x, tmp_y.x);
x[gid].y = 0.0;




};
"""
# scalar_arg_dtypes=[complex64, None, None]