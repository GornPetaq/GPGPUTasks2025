#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

uint get_current_radix(uint target, uint current_radix)
{
    return (target >> (RADIX_SIZE * current_radix)) & ((1 << RADIX_SIZE) - 1);
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* input,
    __global const uint* partial,
    __global uint* output,
    unsigned int radixid,
    unsigned int n,
    unsigned int sum_levels
    ,     __global uint* debug_buf
    )
{
    uint g = get_group_id(0);
    uint l = get_local_id(0);
    uint i = get_global_id(0);
    uint res = 0;
    const uint num_bins = 1 << RADIX_SIZE;
    __local uint chunk[GROUP_SIZE];

    uint my_number_tmp = ~0;
    if (i < n)
        my_number_tmp = input[i];

    const uint my_number = my_number_tmp;
    chunk[l] = my_number;

    barrier(CLK_LOCAL_MEM_FENCE);

    // uint binsval [numbins];
    // for (int k = 0; k < numbins;k++) binsval[numbins]  = 0;

    uint count = 0;
    uint prevgroupcount = 0;

    const uint my_radix = get_current_radix(my_number, radixid);
    for (uint j = 0; j < l; j++) {
        if (my_radix == get_current_radix(chunk[j], radixid))
            count += 1;
    }

    // for (uint j = l + 1; j < GROUP_SIZE; j++) {
    //     if (my_radix > get_current_radix (chunk[j], radixid)) count += 1;
    // }

    // const uint num_of_prev_groups = ((n - 1) / GROUP_SIZE + 1) * (my_radix) + g;
    const uint num_of_prev_groups = ((n - 1) / GROUP_SIZE + 1) * (my_radix) + g;
    prevgroupcount = 0;
    if (num_of_prev_groups >= 1) {

    prevgroupcount = debug_buf[num_of_prev_groups - 1];
    }

    // uint cur_num_of_groups = num_of_prev_groups;
    // uint cur_offset = 0;

    // for (uint k = 0; k < sum_levels - 1; k++) {
    //     if (cur_num_of_groups > 0) {
    //         for (uint j = 0; j < cur_num_of_groups % GROUP_SIZE; j++)
    //             prevgroupcount += partial[cur_offset + cur_num_of_groups - 1 - j];
    //     }

    //     cur_offset += cur_num_of_groups;
    //     cur_num_of_groups /= GROUP_SIZE;
    // }

    // for (uint j = 0; j < cur_num_of_groups; j++) {
    //     prevgroupcount += partial[cur_offset + j];
    // }

    // buffer2 [l * ((n - 1) / GROUP_SIZE + 1) + g] = count;
    if (i < n) {
        output[count + prevgroupcount] = my_number;

        // if (l == 0)
        // debug_buf[num_of_prev_groups] = prevgroupcount;

        // if (radixid == 0) {
        //     printf ("pos last: %d, pos next: %d, real number: %x, my_number: %x\n",i, count , input[i], my_number);
        // }
    }
}