#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* buffer1,
    __global       uint* buffer2,
    unsigned int radix,
    uint n)
{
    uint g = get_group_id(0);
    uint l = get_local_id(0);
    uint i = get_global_id(0);
    uint res = 0;
    const uint num_bins = 1 << RADIX_SIZE;
    __local uint chunk [GROUP_SIZE];
    if (i < n) chunk [l] = buffer1[i];
    else chunk[l] = ~0;

    barrier(CLK_LOCAL_MEM_FENCE);

    // uint binsval [numbins];
    // for (int k = 0; k < numbins;k++) binsval[numbins]  = 0;

    if  (l < num_bins) {
    uint count = 0;
    for (uint j = 0; j < GROUP_SIZE; j++) {
        if (((chunk[j] >> (radix * RADIX_SIZE)) & (num_bins - 1)) == l) count += 1;
    }

    buffer2 [l * ((n - 1) / GROUP_SIZE + 1) + g] = count;
    }


    
    // for (uint j = 0; j < l; j++) {
    //     binsval[(chunk[j] >> (radix * RADIX_SIZE)) & (numbins - 1)] += 1;
    // }

}
