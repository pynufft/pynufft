R="""
KERNEL void cMultiplyRealInplace( 
        const unsigned int batch, 
        GLOBAL_MEM const float *a,
        GLOBAL_MEM float2 *outb)
{
const unsigned int gid = get_global_id(0);
// const unsigned int voxel_id = gid / batch;
const unsigned int voxel_id = (float)gid / (float)batch;
float mul = a[voxel_id];
float2 orig = outb[gid];
orig.x=orig.x*mul; 
orig.y=orig.y*mul;  
outb[gid]=orig;
};
"""
scalar_arg_dtypes=[None, None, None]