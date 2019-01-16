from __future__ import absolute_import # Python2 compatibility

def create_kernel_sets(API):
    kernel_sets = ( cMultiplyScalar() + 
                                cCopy() + cTensorCopy() + cHypot.R +cTensorMultiply.R + 
                        cAddScalar.R + 
                        cSelect.R + 
                        cMultiplyConjVec.R + 
                        cAddVec.R+  
                        cMultiplyVecInplace.R + cMultiplyConjVecInplace.R +cMultiplyRealInplace.R + 
                        cDiff.R+ cSqrt.R+ cAnisoShrink.R+ cMultiplyVec.R + cSpmv.R + cSpmvh.R + cHadamard.R)
    if 'cuda' is API:
        print('Select cuda interface')
        kernel_sets =  atomic_add.cuda_add + kernel_sets
    elif 'ocl' is API:
        print("Selecting opencl interface")
        kernel_sets =  atomic_add.ocl_add + kernel_sets
    return kernel_sets

def cMultiplyScalar():
    """
    cMultiplyScalar subroutine.    
    """
    
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
    return R

def cCopy():
    R="""
            KERNEL void cCopy( 
                     GLOBAL_MEM  const float2 *CX,
                     GLOBAL_MEM             float2 *CY)
            {
            // Copy x to y: y = x;
            //CX: input array (float2)
            // CY output array (float2)
            int gid=get_global_id(0);  
            CY[gid]=CX[gid];
            };
            """  
    return R

def cTensorCopy():
    R="""
    KERNEL void cTensorCopy(
        const uint batch, 
        const uint dim,
        GLOBAL_MEM const  uint *Nd_elements,
        GLOBAL_MEM const  uint *Kd_elements,
        GLOBAL_MEM const  float *invNd,
        GLOBAL_MEM const float2 *indata,
        GLOBAL_MEM       float2 *outdata,
        const int direction)
    {
    
    const uint gid=get_global_id(0); 
    
    uint curr_res = gid;
    uint new_idx = 0;
    uint group;
    
    for (uint dimid =0; dimid < dim; dimid ++){
        group = (float)curr_res*invNd[dimid];
        new_idx += group * Kd_elements[dimid];
        curr_res = curr_res - group * Nd_elements[dimid];
    };
    
    if (direction == 1) {
        for (uint bat=0; bat < batch; bat ++ )
        {
            outdata[new_idx*batch+bat]= indata[gid*batch+bat];
         };   
    };
    
    if (direction == -1) {
        for (uint bat=0; bat < batch; bat ++ )
        {
            outdata[gid*batch+bat]= indata[new_idx*batch+bat];
        };   
    };
                   
                   
    };
    """  
    return R
