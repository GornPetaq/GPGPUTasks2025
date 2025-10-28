#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* inp,
    __global       uint* out,
    unsigned int sz0,
    unsigned int sum_levels)
{

    
    uint g = get_group_id(0);
    uint l = get_local_id(0);
    uint i = get_global_id(0);

    if (i > sz0) return;

    uint sum = 0;

    uint currsz = sz0 - 1;
    uint offset = 0;
    uint curri = i + 1;

    for (uint k = 0; k < sum_levels; k++) {
        uint start = 0;
        uint end = curri;
        if (k != sum_levels - 1) {
            start = curri - (curri % GROUP_SIZE);
        }
        // if (i == 65535) {
        //     printf ("summing from %d to %d, current offset is %d\n", offset + start, offset + end, offset); 
        // }
        for (uint j = start; j < end; j++) {sum += inp[offset + j];
        }

        offset += currsz + 1;
        currsz /= GROUP_SIZE;
        curri /= GROUP_SIZE;
    }

    out[i] = sum;

}
