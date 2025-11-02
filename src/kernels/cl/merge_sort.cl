#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
merge_sort(
    __global const uint* input_data,
    __global uint* output_data,
    int already_sorted,
    int n,
    int launch_type)
{
    const unsigned int i = get_global_id(0);
    const unsigned int l = get_local_id(0);

    bool active = i < n;

    // if (i >= n)
    //     return;

    uint my_num = input_data[i];

    int left = i & (~(already_sorted * 2 - 1));
    // int right = left + 2 * already_sorted;

    // if (left + 2 * already_sorted > n) {
    //     printf ("launch type = %d, already_sorted = %d, n = %d, left + 2*as = %d\n", launch_type,already_sorted, n, left + 2 * already_sorted );
    // }

    int right = min(left + 2 * already_sorted, n);
    // int mid = left + already_sorted;
    int mid = min(left + already_sorted, n);

    __local uint reference_values[GROUP_SIZE];
    const int delta = already_sorted * 2 / GROUP_SIZE;
    // if (l != 0) {
        reference_values[l] = input_data[left + delta * (l + 1) - 1];
    // }

    barrier (CLK_LOCAL_MEM_FENCE);

    bool is_from_left = i < mid;

    int s_left = (is_from_left ? mid : left) - 1;
    int s_right = is_from_left ? right : mid;

    while (s_right > s_left + 1) {
        int s_mid = (s_left + s_right) / 2;
        // if (i == 0 && already_sorted == 1) {
        //     printf("s_left = %d, s_right = %d, s_mid = %d\n", s_left, s_right, s_mid);
        // }
        // if (((s_mid - left + 1) & (delta - 1)) == 0) {

        // }

        // if ((((s_mid - left + 1) & (delta - 1)) == 0)) {
        //     if (reference_values[((s_mid - left + 1) / (delta)) - 1] != input_data[s_mid]) {
        //         printf ("sdfs %d %d \n", ((s_mid - left + 1) / (delta) - 1), s_mid);
        //     }
        // }
        // uint v_mid = (((s_mid - left + 1) & (delta - 1)) == 0) ? reference_values[((s_mid - left + 1) / (delta)) - 1] : input_data[s_mid];
        uint v_mid = input_data[s_mid];
        bool is_less = is_from_left ? (my_num > v_mid) : (my_num >= v_mid);

        if (is_less) {
            s_left = s_mid;
        } else
            s_right = s_mid;
        // if (i == 0 && already_sorted == 1) {
        //     printf("s_left = %d, s_right = %d, s_mid = %d\n", s_left, s_right, s_mid);
        // }
    }

    int idx = (i + s_right - mid);

    if (launch_type == 0 || active)
        output_data[idx] = my_num;

    // if (i == 0 && already_sorted == 1) {
    //     printf("i = %d, v = %d, oi = %d, is_from_left = %d, ast = %d\n", i, my_num, idx, is_from_left, already_sorted);
    // }

    // TODO
}
