// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "bench/utils.h"
#include "src/xnnpack/aligned-allocator.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/ibilinear.h"
#include "src/xnnpack/microfnptr.h"
#include "test/replicable_random_device.h"

static void f32_ibilinear_chw(benchmark::State& state,
                              xnn_f32_ibilinear_chw_ukernel_fn ibilinear,
                              uint64_t arch_flags = 0) {
  if (!benchmark::utils::CheckArchFlags(state, arch_flags)) {
    return;
  }

  const size_t channels = state.range(0);
  const size_t output_pixels = state.range(1);
  // Per-channel input stride, matching the microkernel tester's default.
  const size_t input_stride = 4 * output_pixels + 16;

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist;

  // Indirection points to the even ("left") pixels; kernels expect the "right"
  // pixel right next to each. See IBilinearMicrokernelTester::TestCHW.
  xnnpack::Buffer<const float*> indirection(output_pixels * 2);
  xnnpack::Buffer<float> input((channels - 1) * input_stride +
                               4 * output_pixels);
  xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> packed_weights(
      output_pixels * 2);
  xnnpack::Buffer<float> output(output_pixels * channels);

  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
  std::generate(packed_weights.begin(), packed_weights.end(),
                [&]() { return f32dist(rng); });
  for (size_t i = 0; i < indirection.size(); i++) {
    indirection[i] = input.data() + 2 * i;
  }

  benchmark::utils::PerfCounters perf;
  perf.Start();
  for (auto _ : state) {
    ibilinear(output_pixels, channels, indirection.data(), /*input_offset=*/0,
              packed_weights.data(), output.data(),
              input_stride * sizeof(float));
  }
  perf.Stop();
  perf.Report(state);

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const uint64_t elements_per_iteration = output_pixels * channels;
  state.counters["elements"] = benchmark::Counter(
      static_cast<double>(state.iterations() * elements_per_iteration),
      benchmark::Counter::kIsRate);
}

// A few CHW image-upsampling shapes: {channels, output_pixels}.
static void BilinearArguments(benchmark::Benchmark* b) {
  b->ArgNames({"channels", "pixels"});
  b->Args({16, 4096});  // plan ~16 KB = multiple de 8 KB -> aliase
  b->Args({16, 4000});  // plan ~15.6 KB -> n'aliase plus->Args({32, 32 * 32});
  b->Args({4, 32 * 32});
  b->Args({8, 32 * 32});
  b->Args({16, 32 * 32});
  b->Args({32, 32 * 32});
}

#define BENCHMARK_IBILINEAR_CHW(ukernel, arch_flags)                 \
  BENCHMARK_CAPTURE(f32_ibilinear_chw, ukernel, ukernel, arch_flags) \
      ->Apply(BilinearArguments)                                     \
      ->UseRealTime();

BENCHMARK_IBILINEAR_CHW(xnn_f32_ibilinear_chw_ukernel__scalar_p2, 0);
#if XNN_ARCH_RISCV
BENCHMARK_IBILINEAR_CHW(xnn_f32_ibilinear_chw_ukernel__rvv_u2v_dev,
                        xnn_arch_riscv_vector);
BENCHMARK_IBILINEAR_CHW(xnn_f32_ibilinear_chw_ukernel__rvv_u2v_inv,
                        xnn_arch_riscv_vector);
#endif  // XNN_ARCH_RISCV

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
