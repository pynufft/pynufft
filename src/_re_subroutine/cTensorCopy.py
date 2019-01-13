R="""
KERNEL void cTensorCopy(
    const unsigned int batch, 
    const unsigned int dim,
    GLOBAL_MEM const  unsigned int *Nd_elements,
    GLOBAL_MEM const  unsigned int *Kd_elements,
    GLOBAL_MEM const  float *invNd,
    GLOBAL_MEM const float2 *indata,
    GLOBAL_MEM       float2 *outdata,
    const int direction)
{

const unsigned int gid=get_global_id(0); 

unsigned int curr_res = gid;
unsigned int new_idx = 0;
unsigned int group;

for (unsigned int dimid =0; dimid < dim; dimid ++){
    group = (float)curr_res*invNd[dimid];
    new_idx += group * Kd_elements[dimid];
    curr_res = curr_res - group * Nd_elements[dimid];
};

if (direction == 1) {
    for (unsigned int bat=0; bat < batch; bat ++ )
    {
        outdata[new_idx*batch+bat]= indata[gid*batch+bat];
     };   
};

if (direction == -1) {
    for (unsigned int bat=0; bat < batch; bat ++ )
    {
        outdata[gid*batch+bat]= indata[new_idx*batch+bat];
    };   
};
               
               
};
"""  
