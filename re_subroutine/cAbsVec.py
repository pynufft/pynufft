R="""
KERNEL void cAbsVec( GLOBAL_MEM const ${ctype} *indata, 
                                    GLOBAL_MEM            ${ctype} *outdata)
{
    const int gid =  get_global_id(0);
    ${ctype} tmp = indata[gid];
    tmp.x = sqrt( tmp.x*tmp.x+tmp.y*tmp.y);
    //tmp.x =  sqrt(tmp.x);
    tmp.y = 0.0;
    outdata[gid]=tmp;
};
"""
scalar_arg_dtypes=[None, None]