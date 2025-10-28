#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/fast_random.h>
#include <libbase/timer.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include "debug.h" // TODO очень советую использовать debug::prettyBits(...) для отладки

#include <fstream>
#include <iomanip>

void run(int argc, char** argv)
{
    // chooseGPUVkDevices:
    // - Если не доступо ни одного устройства - кинет ошибку
    // - Если доступно ровно одно устройство - вернет это устройство
    // - Если доступно N>1 устройства:
    //   - Если аргументов запуска нет или переданное число не находится в диапазоне от 0 до N-1 - кинет ошибку
    //   - Если аргумент запуска есть и он от 0 до N-1 - вернет устройство под указанным номером
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);

    // TODO 000 сделайте здесь свой выбор API - если он отличается от OpenCL то в этой строке нужно заменить TypeOpenCL на TypeCUDA или TypeVulkan
    // TODO 000 после этого изучите этот код, запустите его, изучите соответсвующий вашему выбору кернел - src/kernels/<ваш выбор>/aplusb.<ваш выбор>
    // TODO 000 P.S. если вы выбрали CUDA - не забудьте установить CUDA SDK и добавить -DCUDA_SUPPORT=ON в CMake options
    // TODO 010 P.S. так же в случае CUDA - добавьте в CMake options (НЕ меняйте сами CMakeLists.txt чтобы не менять окружение тестирования):
    // TODO 010 "-DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS=-lineinfo" (первое - чтобы включить поддержку WMMA, второе - чтобы compute-sanitizer и профилировщик знали номера строк кернела)
    gpu::Context context = activateContext(device, gpu::Context::TypeOpenCL);
    // OpenCL - рекомендуется как вариант по умолчанию, можно выполнять на CPU, есть printf, есть аналог valgrind/cuda-memcheck - https://github.com/jrprice/Oclgrind
    // CUDA   - рекомендуется если у вас NVIDIA видеокарта, есть printf, т.к. в таком случае вы сможете пользоваться профилировщиком (nsight-compute) и санитайзером (compute-sanitizer, это бывший cuda-memcheck)
    // Vulkan - не рекомендуется, т.к. писать код (compute shaders) на шейдерном языке GLSL на мой взгляд менее приятно чем в случае OpenCL/CUDA
    //          если же вас это не останавливает - профилировщик (nsight-systems) при запуске на NVIDIA тоже работает (хоть и менее мощный чем nsight-compute)
    //          кроме того есть debugPrintfEXT(...) для вывода в консоль с видеокарты
    //          кроме того используемая библиотека поддерживает rassert-проверки (своеобразные инварианты с уникальным числом) на видеокарте для Vulkan

    ocl::KernelSource ocl_fillBufferWithZeros(ocl::getFillBufferWithZeros());
    ocl::KernelSource ocl_radixSort01AccumulateToSingleBuf(ocl::getRadixSort01LocalCounting());
    ocl::KernelSource ocl_radixSort02GlobalPrefixesScanSumReduction(ocl::getRadixSort02GlobalPrefixesScanSumReduction());
    ocl::KernelSource ocl_radixSort03GlobalPrefixesScanAccumulation(ocl::getRadixSort03GlobalPrefixesScanAccumulation());
    ocl::KernelSource ocl_radixSort04Scatter(ocl::getRadixSort04Scatter());

    avk2::KernelSource vk_fillBufferWithZeros(avk2::getFillBufferWithZeros());
    avk2::KernelSource vk_radixSort01LocalCounting(avk2::getRadixSort01LocalCounting());
    avk2::KernelSource vk_radixSort02GlobalPrefixesScanSumReduction(avk2::getRadixSort02GlobalPrefixesScanSumReduction());
    avk2::KernelSource vk_radixSort03GlobalPrefixesScanAccumulation(avk2::getRadixSort03GlobalPrefixesScanAccumulation());
    avk2::KernelSource vk_radixSort04Scatter(avk2::getRadixSort04Scatter());

    FastRandom r;

    int n = 100 * 1000 * 1000; // TODO при отладке используйте минимальное n (например n=5 или n=10) при котором воспроизводится бага
    // int n = 10000; // TODO при отладке используйте минимальное n (например n=5 или n=10) при котором воспроизводится бага
    int max_value = std::numeric_limits<int>::max(); // TODO при отладке используйте минимальное max_value (например max_value=8) при котором воспроизводится бага
    std::vector<unsigned int> as(n, 0);
    std::vector<unsigned int> sorted(n, 0);
    for (size_t i = 0; i < n; ++i) {
        as[i] = r.next(0, max_value);
    }
    std::cout << "n=" << n << " max_value=" << max_value << std::endl;

    {
        // убедимся что в массиве есть хотя бы несколько повторяющихся значений
        size_t force_duplicates_attempts = 3;
        bool all_attempts_missed = true;
        for (size_t k = 0; k < force_duplicates_attempts; ++k) {
            size_t i = r.next(0, n - 1);
            size_t j = r.next(0, n - 1);
            if (i != j) {
                as[j] = as[i];
                all_attempts_missed = false;
            }
        }
        rassert(!all_attempts_missed, 4353245123412);
    }

    {
        sorted = as;
        std::cout << "sorting on CPU..." << std::endl;
        timer t;
        std::sort(sorted.begin(), sorted.end());
        // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили его переупорядоченным)
        double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
        std::cout << "CPU std::sort finished in " << t.elapsed() << " sec" << std::endl;
        std::cout << "CPU std::sort effective RAM bandwidth: " << memory_size_gb / t.elapsed() << " GB/s (" << n / 1000 / 1000 / t.elapsed() << " uint millions/s)" << std::endl;
    }

    // Аллоцируем буферы в VRAM
    gpu::gpu_mem_32u input_gpu(n);

    std::vector <int> sizes;
    std::vector <int> bufedges {0,};
    const int sum_levels = 3;
    int tn = ((n - 1) / GROUP_SIZE + 1) * ( 1 << RADIX_SIZE) - 1;
    for (int i = 0; i < sum_levels; i++) {
        sizes.push_back(tn + 1);
        tn /= GROUP_SIZE;
        bufedges.push_back(bufedges.back() + sizes.back());
    }

    

    gpu::gpu_mem_32u buffer1_gpu(n), buffer2_gpu(bufedges.back()), buffer3_gpu(n), debug_buf(n); // TODO это просто шаблонка, можете переименовать эти буферы, сделать другого размера/типа, удалить часть, добавить новые
    gpu::gpu_mem_32u buffer_output_gpu(n);

    // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
    input_gpu.writeN(as.data(), n);
    // Советую занулить (или еще лучше - заполнить какой-то уникальной константой, например 255) все буферы
    // В некоторых случаях это ускоряет отладку, но обратите внимание, что fill реализован через копию множества нулей по PCI-E, то есть он очень медленный
    // Если вам нужно занулять буферы в процессе вычислений - используйте кернел который это сделает (см. кернел fill_buffer_with_zeros)
    buffer1_gpu.fill(255);
    buffer2_gpu.fill(255);
    buffer3_gpu.fill(255);
    debug_buf.fill(254);
    buffer_output_gpu.fill(255);

    // Запускаем кернел (несколько раз и с замером времени выполнения)
    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) { // TODO при отладке запускайте одну итерацию
        timer t;

        // Запускаем кернел, с указанием размера рабочего пространства и передачей всех аргументов
        // Если хотите - можете удалить ветвление здесь и оставить только тот код который соответствует вашему выбору API
        if (context.type() == gpu::Context::TypeOpenCL) {

            
            // TODO
            // if (iter == 1) {
            //     for (auto i : input_gpu.readVector(n))
            //         std::cout << std::hex << i << " ";
            //     std::cout << "\n";
            //     std::cout.flush();
            // }

            auto sort_radix = [&](gpu::gpu_mem_32u& input, gpu::gpu_mem_32u& intermed, gpu::gpu_mem_32u& output, unsigned radixid) {
                ocl_radixSort02GlobalPrefixesScanSumReduction.exec(gpu::WorkSize(GROUP_SIZE, n), input, intermed, radixid, n);
                for (int i = 1; i < sum_levels; i++) {
                    ocl_radixSort03GlobalPrefixesScanAccumulation.exec (gpu::WorkSize(GROUP_SIZE, sizes[i - 1]), intermed, bufedges[i-1], intermed, bufedges[i], sizes[i - 1]);
                }

                ocl_radixSort01AccumulateToSingleBuf.exec(gpu::WorkSize(GROUP_SIZE, sizes[0]), intermed, debug_buf, sizes[0], sum_levels);


                ocl_radixSort04Scatter.exec(gpu::WorkSize(GROUP_SIZE, n), input, intermed, output, radixid, n, sum_levels, debug_buf);

                auto check_correct = [&]() -> void {
                    auto imed = intermed.readVector();
                    for (int k = 0; k < sum_levels - 1; k++) {
                    for (int i = 0; i < sizes[k + 1]; i++) {
                        int sum = 0;
                        for (int j = 0; j < GROUP_SIZE && (i * GROUP_SIZE + j < sizes[k] ); j++) {
                            sum += imed[bufedges[k] + i * GROUP_SIZE + j];
                        }

                        if (sum != imed[bufedges[k + 1] + i]) throw std::runtime_error (to_string("check_correct failed at radixid ") +
                         to_string(radixid) + " k = " + to_string(k) + " i = " + to_string(i));
                    }
                    }

                    std::cout << bufedges[2] << " "  << imed [bufedges[2]] << " sdhlaod\n";

                    auto dbuf = debug_buf.readVector();
                    int totsum = 0;
                    for (int i = 0; i < sizes[0];i++) {
                        totsum += imed[i];
                        if (totsum != dbuf[i]) throw std::runtime_error (to_string("check_correct failed at radixid ") +
                         to_string(radixid) + " totsum = " + to_string(totsum) + " i = " + to_string(i) + " dbuf[i] = " + to_string(dbuf[i]));
                    }



                };

                // check_correct();

                // if (iter == 1) {
                //     std::cout << "radixid: "<<radixid << "\n";
                // // auto res = intermed.readVector( ((n- 1) / GROUP_SIZE + 1) * ( 1 << RADIX_SIZE));
                // auto res = output.readVector(n);
                //     for (auto i : res)
                //         std::cout << std::hex << i << " ";
                //     std::cout << "\n";
                // }
            };

            sort_radix(input_gpu, buffer2_gpu, buffer1_gpu, 0);
            for (int j = 1; j < (32 / RADIX_SIZE) - 1; j++) {
                if (j % 2 == 1) {
                    sort_radix(buffer1_gpu, buffer2_gpu, buffer3_gpu, j);
                } else {
                    sort_radix(buffer3_gpu, buffer2_gpu, buffer1_gpu, j);
                }
            }

            if (int j = (32 / RADIX_SIZE) - 1; j % 2 == 1) {
                sort_radix(buffer1_gpu, buffer2_gpu, buffer_output_gpu, j);
            } else {
                sort_radix(buffer3_gpu, buffer2_gpu, buffer_output_gpu, j);
            }

            // assume j is odd at the end

            // throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
            // ocl_fillBufferWithZeros.exec();
            // ocl_radixSort01LocalCounting.exec();
            // ocl_radixSort02GlobalPrefixesScanSumReduction.exec();
            // ocl_radixSort03GlobalPrefixesScanAccumulation.exec();
            // ocl_radixSort04Scatter.exec();
        } else if (context.type() == gpu::Context::TypeCUDA) {
            // TODO
            throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
            // cuda::fill_buffer_with_zeros();
            // cuda::radix_sort_01_local_counting();
            // cuda::radix_sort_02_global_prefixes_scan_sum_reduction();
            // cuda::radix_sort_03_global_prefixes_scan_accumulation();
            // cuda::radix_sort_04_scatter();
        } else if (context.type() == gpu::Context::TypeVulkan) {
            // TODO
            throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
            // vk_fillBufferWithZeros.exec();
            // vk_radixSort01LocalCounting.exec();
            // vk_radixSort02GlobalPrefixesScanSumReduction.exec();
            // vk_radixSort03GlobalPrefixesScanAccumulation.exec();
            // vk_radixSort04Scatter.exec();
        } else {
            rassert(false, 4531412341, context.type());
        }

        times.push_back(t.elapsed());
    }
    std::cout << "GPU radix-sort times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

    // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили его переупорядоченным)
    double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
    std::cout << "GPU radix-sort median effective VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s (" << n / 1000 / 1000 / stats::median(times) << " uint millions/s)" << std::endl;

    // Считываем результат по PCI-E шине: GPU VRAM -> CPU RAM
    std::vector<unsigned int> gpu_sorted = buffer_output_gpu.readVector();

    // Сверяем результат
    for (size_t i = 0; i < n; ++i) {
        rassert(sorted[i] == gpu_sorted[i], 566324523452323, sorted[i], gpu_sorted[i], i);
    }

    // Проверяем что входные данные остались нетронуты (ведь мы их переиспользуем от итерации к итерации)
    std::vector<unsigned int> input_values = input_gpu.readVector();
    for (size_t i = 0; i < n; ++i) {
        rassert(input_values[i] == as[i], 6573452432, input_values[i], as[i]);
    }
}

int main(int argc, char** argv)
{
    try {
        run(argc, argv);
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        if (e.what() == DEVICE_NOT_SUPPORT_API) {
            // Возвращаем exit code = 0 чтобы на CI не было красного крестика о неуспешном запуске из-за выбора CUDA API (его нет на процессоре - т.е. в случае CI на GitHub Actions)
            return 0;
        }
        if (e.what() == CODE_IS_NOT_IMPLEMENTED) {
            // Возвращаем exit code = 0 чтобы на CI не было красного крестика о неуспешном запуске из-за того что задание еще не выполнено
            return 0;
        } else {
            // Выставляем ненулевой exit code, чтобы сообщить, что случилась ошибка
            return 1;
        }
    }

    return 0;
}
