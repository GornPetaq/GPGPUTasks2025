#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* in,
    unsigned int in_start,
    __global       uint* out,
    unsigned int out_start,
    unsigned int in_n)
{
    __local uint chunk [GROUP_SIZE];

    uint g = get_group_id(0);
    uint l = get_local_id(0);
    uint i = get_global_id(0);

    if (i < in_n) chunk[l] = in[in_start + i];
    else chunk[l] = 0;

    barrier (CLK_LOCAL_MEM_FENCE);

    if (l == 0) {
        uint res = 0 ;
        for (int k = 0; k < GROUP_SIZE; k++) {
            res += chunk[k];
        }

        out[out_start + g] = res;
    }

}
