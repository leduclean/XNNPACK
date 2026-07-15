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



void xnn_f32_ibilinear_chw_ukernel__rvv_u2v_pack(
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

      // Each (alphah, alphav) pair occupies 8 contiguous bytes, i.e. one u64
      // lane: a unit-stride vle64 replaces the element-wise vlseg2e32, and
      // the halves are split by two narrowing shifts.
      vuint64m4_t vwpair =
          __riscv_vle64_v_u64m4((const uint64_t*)w, vl);
      vfloat32m2_t vwh = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vnsrl_wx_u32m2(vwpair, 0, vl));
      vfloat32m2_t vwv = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vnsrl_wx_u32m2(vwpair, 32, vl));

      // top_left and top_right are adjacent in memory, so a single 64-bit
      // gathered lane carries both -- the RVV analogue of NEON's vld1_f32().
      // Halves the gathered element count versus vluxseg2ei64.
      vuint64m4_t vtpair = __riscv_vluxei64_v_u64m4(
          (const uint64_t*)input_offset, vti_addr, vl);
      vfloat32m2_t vtl = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vnsrl_wx_u32m2(vtpair, 0, vl));
      vfloat32m2_t vtr = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vnsrl_wx_u32m2(vtpair, 32, vl));

      vuint64m4_t vbpair = __riscv_vluxei64_v_u64m4(
          (const uint64_t*)input_offset, vbi_addr, vl);
      vfloat32m2_t vbl = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vnsrl_wx_u32m2(vbpair, 0, vl));
      vfloat32m2_t vbr = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vnsrl_wx_u32m2(vbpair, 32, vl));

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
