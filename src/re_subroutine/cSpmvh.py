"""
==============================================
Conjugate transpose of spmv.
Offload the conjugate transpose of Sparse Matrix Vector Multiplication to heterogeneous devices.
Note: Now pyopencl/pycuda are supported. 
"""

R="""
KERNEL void cCSR_spmvh_scalar(   
        const    uint    n_row,
        GLOBAL_MEM    const    uint    *indptr, 
        GLOBAL_MEM    const    uint    *indices,
        GLOBAL_MEM    const    float2    *data,
        GLOBAL_MEM    float    *kx, 
        GLOBAL_MEM    float    *ky,
        GLOBAL_MEM   const    float2    *Yx)
{   
    uint myRow = get_global_id(0);
    float2 zero;
    zero.x=0.0;
    zero.y=0.0;
    float2  u = zero;
    if (myRow < n_row)
    {   
        uint vecStart = indptr[myRow];
        uint vecEnd = indptr[myRow+1];
        float2 y =  Yx[myRow];
        for (uint j = vecStart + 0; j < vecEnd;  j += 1) //vecWidth)
        {
            const uint col = indices[j];
            const float2 spdata = data[j]; // row
            u.x =  spdata.x * y.x + spdata.y * y.y;
            u.y =  - spdata.y * y.x + spdata.x * y.y;
            atomic_add_float(kx + col, u.x);
            atomic_add_float(ky + col, u.y);
        }; // for uint j 
    }; //  if (myRow < n_row)
}; // Program entry
                
KERNEL void cELL_spmvh_scalar(   
        const    uint    n_row,
        GLOBAL_MEM    const    uint    *indptr, 
        GLOBAL_MEM    const    uint    *indices,
        GLOBAL_MEM    const    float2    *data,
        GLOBAL_MEM    float    *kx, 
        GLOBAL_MEM    float    *ky,
        GLOBAL_MEM   const    float2    *Yx)
{   
    uint myRow = get_global_id(0);
    float2 zero;
    zero.x=0.0;
    zero.y=0.0;
    float2  u = zero;
    if (myRow < n_row)
    {   
        uint vecStart = indptr[myRow];
        uint vecEnd = indptr[myRow+1];
        float2 y =  Yx[myRow];
        for (uint j = vecStart + 0; j < vecEnd;  j += 1) //vecWidth)
        {
            const uint col = indices[j];
            const float2 spdata = data[j]; // row
            u.x =  spdata.x * y.x + spdata.y * y.y;
            u.y =  - spdata.y * y.x + spdata.x * y.y;
            atomic_add_float(kx + col, u.x);
            atomic_add_float(ky + col, u.y);
        }; // for uint j 
    }; //  if (myRow < n_row)
}; // Program entry    
    
KERNEL void pELL_spmvh_scalar(    
        const    uint    nRow,        // number of rows
        const    uint    prodJd,     // product of Jd
        const    uint    sumJd,     // sum of Jd
        const    uint    dim,           // dimensionality
        GLOBAL_MEM const uint *Jd,    // Jd
        // GLOBAL_MEM const uint *curr_sumJd,    // 
        GLOBAL_MEM const uint *meshindex,    // meshindex, prodJd * dim
        GLOBAL_MEM const uint *kindx,    // unmixed column indexes of all dimensions
        GLOBAL_MEM const float2 *udata,    // interpolation data before Kronecker product
        GLOBAL_MEM    float    *kx, 
        GLOBAL_MEM    float    *ky,
        GLOBAL_MEM const float2 *input)   // y
{      
    uint myRow= get_global_id(0);
    float2 zero;
    zero.x = 0.0;
    zero.y = 0.0;
    if (myRow < nRow)
    {
         float2 u = zero;
         for (uint j = 0;  j  <  prodJd; j ++)
         {    
            // now doing the first dimension
            uint index_shift = myRow * sumJd;
            // uint tmp_sumJd = 0;
            uint J = Jd[0];
            uint index =    index_shift +  meshindex[dim*j + 0];
            uint col = kindx[index] ;
            float2 spdata = udata[index];
            index_shift += J; 
            for (uint dimid = 1; dimid < dim; dimid ++ )
            {
                J = Jd[dimid];
                index =   index_shift + meshindex[dim*j + dimid];   // the index of the partial ELL arrays *kindx and *udata
                col += kindx[index] + 1  ;                                            // the column index of the current j
                float tmp_x = spdata.x;
                float2 tmp_udata = udata[index];
                spdata.x = spdata.x * tmp_udata.x - spdata.y * tmp_udata.y;                            // the spdata of the current j
                spdata.y = tmp_x * tmp_udata.y + spdata.y * tmp_udata.x; 
                index_shift  += J;
            }; // Iterate over dimensions 1 -> Nd - 1
            
            float2 ydata=input[myRow]; // kout[col];
            u.x =  spdata.x*ydata.x + spdata.y*ydata.y;
            u.y =  - spdata.y*ydata.x + spdata.x*ydata.y;
            
            atomic_add_float(kx + col, u.x);
            atomic_add_float(ky + col, u.y);
           
        }; // Iterate for (uint j = 0;  j  <  prodJd; j ++)
    };  // if (myRow < nRow)
};    // End of pELL_spmv_scalar    
    

KERNEL void pELL_spmvh_vector(    
        const    uint    nRow,        // number of rows
        const    uint    prodJd,     // product of Jd
        const    uint    sumJd,     // sum of Jd
        const    uint    dim,           // dimensionality
        GLOBAL_MEM const uint *Jd,    // Jd
        // GLOBAL_MEM const uint *curr_sumJd,    // 
        GLOBAL_MEM const uint *meshindex,    // meshindex, prodJd * dim
        GLOBAL_MEM const uint *kindx,    // unmixed column indexes of all dimensions
        GLOBAL_MEM const float2 *udata,    // interpolation data before Kronecker product
        GLOBAL_MEM    float    *kx, 
        GLOBAL_MEM    float    *ky,
        GLOBAL_MEM const float2 *input)   // y
{      
    const uint t = get_local_id(0);
    const uint d = get_local_size(0);
    const uint g = get_group_id(0);
    const uint a = g/prodJd;
    const uint b = g - (prodJd*a);
    const uint myRow = a*d + t;
    const uint j = b;
    
    // uint myRow= get_global_id(0);
    float2 zero;
    zero.x = 0.0;
    zero.y = 0.0;
    if (myRow < nRow)
    {
         float2 u = zero;
       //  for (uint j = 0;  j  <  prodJd; j ++)
       if (j<prodJd)
         {    
            // now doing the first dimension
            uint index_shift = myRow * sumJd;
            // uint tmp_sumJd = 0;
            uint J = Jd[0];
            uint index =    index_shift +  meshindex[dim*j + 0];
            uint col = kindx[index] ;
            float2 spdata = udata[index];
            index_shift += J; 
            for (uint dimid = 1; dimid < dim; dimid ++ )
            {
                J = Jd[dimid];
                index =   index_shift + meshindex[dim*j + dimid];   // the index of the partial ELL arrays *kindx and *udata
                col += kindx[index] + 1  ;                                            // the column index of the current j
                float tmp_x = spdata.x;
                float2 tmp_udata = udata[index];
                spdata.x = spdata.x * tmp_udata.x - spdata.y * tmp_udata.y;                            // the spdata of the current j
                spdata.y = tmp_x * tmp_udata.y + spdata.y * tmp_udata.x; 
                index_shift  += J;
            }; // Iterate over dimensions 1 -> Nd - 1
            
            float2 ydata=input[myRow]; // kout[col];
            u.x =  spdata.x*ydata.x + spdata.y*ydata.y;
            u.y =  - spdata.y*ydata.x + spdata.x*ydata.y;
            
            atomic_add_float(kx + col, u.x);
            atomic_add_float(ky + col, u.y);
           
      }; // Iterate for (uint j = 0;  j  <  prodJd; j ++)
    };  // if (myRow < nRow)
    
};    // End of pELL_spmv_vector  
    

KERNEL void pELL_spmvh_mCoil(
        const    uint    Nc,             // number of coils
        const    uint    nRow,        // number of rows
        const    uint    prodJd,     // product of Jd
        const    uint    sumJd,     // sum of Jd
        const    uint    dim,           // dimensionality
        GLOBAL_MEM const uint *Jd,    // Jd
        // GLOBAL_MEM const uint *curr_sumJd,    // 
        GLOBAL_MEM const uint *meshindex,    // meshindex, prodJd * dim
        GLOBAL_MEM const uint *kindx,    // unmixed column indexes of all dimensions
        GLOBAL_MEM const float2 *udata,    // interpolation data before Kronecker product
        GLOBAL_MEM    float    *kx, 
        GLOBAL_MEM    float    *ky,
        GLOBAL_MEM const float2 *input)   // y
{      
    const uint t = get_local_id(0);
    const uint d = get_local_size(0);
    const uint g = get_group_id(0);
    const uint a = g/prodJd;
    const uint b = g - (prodJd*a);
    const uint myRow = a*d + t;
    const uint j = b;
    uint m = myRow / Nc;
    uint nc = myRow - m * Nc;
    
    // uint myRow= get_global_id(0);
    float2 zero;
    zero.x = 0.0;
    zero.y = 0.0;
    if (myRow < nRow*Nc)
    {
         float2 u = zero;
       //  for (uint j = 0;  j  <  prodJd; j ++)
       if (j<prodJd)
         {    
            // now doing the first dimension
            uint index_shift = m * sumJd;
            // uint tmp_sumJd = 0;
            uint J = Jd[0];
            uint index =    index_shift +  meshindex[dim*j + 0];
            uint col = kindx[index] ;
            float2 spdata = udata[index];
            index_shift += J; 
            for (uint dimid = 1; dimid < dim; dimid ++ )
            {
                J = Jd[dimid];
                index =   index_shift + meshindex[dim*j + dimid];   // the index of the partial ELL arrays *kindx and *udata
                col += kindx[index] + 1  ;                                            // the column index of the current j
                float tmp_x = spdata.x;
                float2 tmp_udata = udata[index];
                spdata.x = spdata.x * tmp_udata.x - spdata.y * tmp_udata.y;                            // the spdata of the current j
                spdata.y = tmp_x * tmp_udata.y + spdata.y * tmp_udata.x; 
                index_shift  += J;
            }; // Iterate over dimensions 1 -> Nd - 1
            
            float2 ydata=input[myRow]; // kout[col];
            u.x =  spdata.x*ydata.x + spdata.y*ydata.y;
            u.y =  - spdata.y*ydata.x + spdata.x*ydata.y;
            
            atomic_add_float(kx + col*Nc + nc, u.x);
            atomic_add_float(ky + col*Nc + nc, u.y);
           
      }; // Iterate for (uint j = 0;  j  <  prodJd; j ++)
    };  // if (myRow < nRow)
    
};    // End of pELL_spmv_mCoil  
        
"""
from numpy import uint32
scalar_arg_dtypes=[uint32, None, None, None, None, None]        
