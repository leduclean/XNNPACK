#include <assert.h>
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/ibilinear.h"

void xnn_f32_ibilinear_chw_ukernel__rvv_u2v_off(
    size_t output_pixels, size_t channels, const float** restrict input,
    size_t input_offset, const float* restrict weights, float* restrict output,
    size_t input_increment) {
  assert(output_pixels != 0);
  assert(channels != 0);
  assert(input_increment % sizeof(float) == 0);

  size_t c = channels;
  vuint64m1_t zero_init = __riscv_vmv_v_x_u64m1(0, 1);
  vuint64m1_t max_init = __riscv_vmv_v_x_u64m1(UINT64_MAX, 1);

  do {
    size_t p = output_pixels;
    const float** i = input;
    const float* w = weights;
    for (size_t vl; p > 0; p -= vl, output += vl, i += 2 * vl, w += 2 * vl) {
      vl = __riscv_vsetvl_e32m2(p);

      vuint64m4x2_t vptrs = __riscv_vlseg2e64_v_u64m4x2((const uint64_t*)i, vl);

      vuint64m4_t vti_addr = __riscv_vget_v_u64m4x2_u64m4(vptrs, 0);
      vuint64m4_t vbi_addr = __riscv_vget_v_u64m4x2_u64m4(vptrs, 1);

      uint64_t min_t = __riscv_vmv_x_s_u64m1_u64(
          __riscv_vredminu_vs_u64m4_u64m1(vti_addr, max_init, vl));
      uint64_t min_b = __riscv_vmv_x_s_u64m1_u64(
          __riscv_vredminu_vs_u64m4_u64m1(vbi_addr, max_init, vl));
      uintptr_t base = min_t < min_b ? min_t : min_b;

      vuint64m4_t vtioff64 = __riscv_vsub_vx_u64m4(vti_addr, base, vl);
      uint64_t maxoffvt = __riscv_vmv_x_s_u64m1_u64(
          __riscv_vredmaxu_vs_u64m4_u64m1(vtioff64, zero_init, vl));

      vuint64m4_t vbioff64 = __riscv_vsub_vx_u64m4(vbi_addr, base, vl);
      uint64_t maxoffvb = __riscv_vmv_x_s_u64m1_u64(
          __riscv_vredmaxu_vs_u64m4_u64m1(vbioff64, zero_init, vl));

      // Stub we consider this is false for now
      bool fallbackvb = maxoffvb > UINT32_MAX;
      bool fallbackvt = maxoffvt > UINT32_MAX;

      vuint32m2_t vtioff32 = __riscv_vncvt_x_x_w_u32m2(vtioff64, vl);
      vuint32m2_t vbioff32 = __riscv_vncvt_x_x_w_u32m2(vbioff64, vl);

      vfloat32m2x2_t wptrs = __riscv_vlseg2e32_v_f32m2x2(w, vl);
      vfloat32m2_t vwh = __riscv_vget_v_f32m2x2_f32m2(wptrs, 0);
      vfloat32m2_t vwv = __riscv_vget_v_f32m2x2_f32m2(wptrs, 1);

      vfloat32m2x2_t vti = __riscv_vluxseg2ei32_v_f32m2x2(
          (const float*)(base + input_offset), vtioff32, vl);
      vfloat32m2_t vtl = __riscv_vget_v_f32m2x2_f32m2(vti, 0);
      vfloat32m2_t vtr = __riscv_vget_v_f32m2x2_f32m2(vti, 1);

      vfloat32m2x2_t vbi = __riscv_vluxseg2ei32_v_f32m2x2(
          (const float*)(base + input_offset), vbioff32, vl);
      vfloat32m2_t vbl = __riscv_vget_v_f32m2x2_f32m2(vbi, 0);
      vfloat32m2_t vbr = __riscv_vget_v_f32m2x2_f32m2(vbi, 1);

      vfloat32m2_t vt = __riscv_vfmacc_vv_f32m2(
          vtl, __riscv_vfsub_vv_f32m2(vtr, vtl, vl), vwh, vl);

      vfloat32m2_t vb = __riscv_vfmacc_vv_f32m2(
          vbl, __riscv_vfsub_vv_f32m2(vbr, vbl, vl), vwh, vl);

      vfloat32m2_t out = __riscv_vfmacc_vv_f32m2(
          vt, __riscv_vfsub_vv_f32m2(vb, vt, vl), vwv, vl);

      __riscv_vse32_v_f32m2(output, out, vl);
    }
    // Go to next channel
    input_offset += input_increment;
    c--;
  } while (c != 0);
}
