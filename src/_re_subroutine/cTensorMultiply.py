R="""
KERNEL void cTensorMultiply(
    const unsigned int batch, // batch 
    const unsigned int dim, // dimensions
    GLOBAL_MEM const  unsigned int *Nd, // In batch mode, Nd*batch but is calculated outside the kernel
    GLOBAL_MEM const  unsigned int *Nd_elements, // Number of elements to move along the dimension = strides / itemsize
    GLOBAL_MEM const  float *invNd_elements,   //  float: inverse of the Nd_elements, which aims for fast division // batch mode: Nd_elements / batch
    GLOBAL_MEM const float *vec, // Real, vector, length sum Nd[dimid]
    GLOBAL_MEM       float2 *outdata, 
    const unsigned int div) 
{
const unsigned int gid=get_global_id(0); 
const unsigned int pid = (float)gid / (float)batch;
// const unsigned int bat = gid - pid * batch;

unsigned int group;
unsigned int Nd_indx_shift = 0;
float mul = 1.0; 
unsigned int res = pid; 

for (unsigned int dimid = 0; dimid < dim; dimid ++){
    group = (float)res * invNd_elements[dimid]; // The index along the axis
    res = res - group * Nd_elements[dimid];
    
    const unsigned int N = Nd[dimid]; 
    
    mul = mul * vec[group + Nd_indx_shift];
    
    Nd_indx_shift = Nd_indx_shift + N;
}

if (div == 1){
    // for (unsigned int bat = 0; bat < batch; bat ++ )
    // {
    float2 tmp = outdata[gid];
    tmp.x = tmp.x /  mul;
    tmp.y = tmp.y / mul;
    outdata[gid] = tmp;
    // };
    };
if (div == 0){
   // for (unsigned int bat = 0; bat < batch; bat ++ )
    // {
    float2 tmp = outdata[gid];
    tmp.x = tmp.x *  mul;
    tmp.y = tmp.y * mul;
    outdata[gid] = tmp;
    // };
    };    
               
};
"""  
