// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-ibilinear-chw/rvv.c.in
//   Generator: tools/xngen.py
//
#include <assert.h>
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/ibilinear.h"


void xnn_f32_ibilinear_chw_ukernel__rvv_u2v(
                size_t output_pixels,
                size_t channels,
                const float** restrict input,
                size_t input_offset,
                const float* restrict weights,
                float* restrict output,
                size_t input_increment) {

  assert(output_pixels != 0);
  assert(channels != 0);
  assert(input_increment % sizeof(float) == 0);

  size_t c = channels;

  do {
    const float** i = input;
    const float* w = weights;
    size_t p = output_pixels;
    for (size_t vl; p > 0; p -= vl, output += vl, i += 2 * vl, w += 2 * vl) {
      vl = __riscv_vsetvl_e32m2(p);

      vuint64m4x2_t vptrs =
          __riscv_vlseg2e64_v_u64m4x2((const uint64_t*)i, vl);

      vuint64m4_t vti_addr =
          __riscv_vget_v_u64m4x2_u64m4(vptrs, 0);
      vuint64m4_t vbi_addr =
          __riscv_vget_v_u64m4x2_u64m4(vptrs, 1);

      vfloat32m2x2_t wptrs = __riscv_vlseg2e32_v_f32m2x2(w, vl);
      vfloat32m2_t vwh =
          __riscv_vget_v_f32m2x2_f32m2(wptrs, 0);
      vfloat32m2_t vwv =
          __riscv_vget_v_f32m2x2_f32m2(wptrs, 1);

      vfloat32m2x2_t vti = __riscv_vluxseg2ei64_v_f32m2x2(
          (const float*)input_offset, vti_addr, vl);
      vfloat32m2_t vtl =
          __riscv_vget_v_f32m2x2_f32m2(vti, 0);
      vfloat32m2_t vtr =
          __riscv_vget_v_f32m2x2_f32m2(vti, 1);

      vfloat32m2x2_t vbi = __riscv_vluxseg2ei64_v_f32m2x2(
          (const float*)input_offset, vbi_addr, vl);
      vfloat32m2_t vbl =
          __riscv_vget_v_f32m2x2_f32m2(vbi, 0);
      vfloat32m2_t vbr =
          __riscv_vget_v_f32m2x2_f32m2(vbi, 1);

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
