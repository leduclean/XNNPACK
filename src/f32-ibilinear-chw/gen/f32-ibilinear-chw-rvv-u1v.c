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


void xnn_f32_ibilinear_chw_ukernel__rvv_u1v(
    size_t output_pixels,
    size_t channels,
    const float** restrict input,
    size_t input_offset,
    const float* restrict weights,
    float* restrict output,
    size_t input_increment) 
{
  assert(output_pixels != 0);
  assert(channels != 0);
  assert(input_increment % sizeof(float) == 0);

  // Gather
  // size_t c = channels;
  // do {
  //   const float** i = input;
  //   const float* w = weights;
  //   size_t p = output_pixels;
  //
  //   for (size_t vl; p > 0;
  //        p -= vl, output += vl, i += 2 * vl, w += 2 * vl) {
  //     vl = __riscv_vsetvl_e32m1(p);
  //     vuint64m2x2_t vptrs = __riscv_vlseg2e64_v_u64m2x2(
  //         (const uint64_t*)i, vl);
  //     vuint64m2_t vti_addr =
  //         __riscv_vget_v_u64m2x2_u64m2(vptrs, 0);
  //     vuint64m2_t vbi_addr =
  //         __riscv_vget_v_u64m2x2_u64m2(vptrs, 1);
  //
  //     vfloat32m1x2_t vti =
  //         __riscv_vluxseg2ei64_v_f32m1x2((const float*)input_offset, vti_addr, vl);
  //     vfloat32m1_t vtl = __riscv_vget_v_f32m1x2_f32m1(vti, 0);
  //     vfloat32m1_t vtr = __riscv_vget_v_f32m1x2_f32m1(vti, 1);
  //
  //     vfloat32m1x2_t vbi = __riscv_vluxseg2ei64_v_f32m1x2((const float*)input_offset, vbi_addr, vl);
  //     vfloat32m1_t vbl = __riscv_vget_v_f32m1x2_f32m1(vbi, 0);
  //     vfloat32m1_t vbr = __riscv_vget_v_f32m1x2_f32m1(vbi, 1);
  //
  //     vfloat32m1x2_t wptrs = __riscv_vlseg2e32_v_f32m1x2(w, vl);
  //     vfloat32m1_t vwh = __riscv_vget_v_f32m1x2_f32m1(wptrs, 0);
  //     vfloat32m1_t vwv = __riscv_vget_v_f32m1x2_f32m1(wptrs, 1);
  //
  //     vfloat32m1_t vt = __riscv_vfmacc_vv_f32m1(
  //         vtl, __riscv_vfsub_vv_f32m1(vtr, vtl, vl), vwh, vl);
  //
  //     vfloat32m1_t vb = __riscv_vfmacc_vv_f32m1(
  //         vbl, __riscv_vfsub_vv_f32m1(vbr, vbl, vl), vwh, vl);
  //
  //     vfloat32m1_t out = __riscv_vfmacc_vv_f32m1(
  //         vt, __riscv_vfsub_vv_f32m1(vb, vt, vl), vwv, vl);
  //
  //     __riscv_vse32_v_f32m1(output, out, vl);
  //   }
  //
  //   // Go to next channel
  //   input_offset += input_increment;
  //   c--;
  // } while (c != 0);
  //stride
  const size_t output_increment = output_pixels * sizeof(float);

  do {
    const float* tl = (const float*) ((uintptr_t) input[0] + input_offset);
    const float* bl = (const float*) ((uintptr_t) input[1] + input_offset);
    input += 2;

    const float valphah = weights[0];   // poids SCALAIRES -> vfmacc_vf
    const float valphav = weights[1];
    weights += 2;

    float* o = output;   // sortie du pixel courant (canal 0)
    output += 1;         // pixel suivant : décalage de 1 (canaux contigus en i)

    size_t c = channels;
    do {
      const size_t n = __riscv_vsetvl_e32m1(c);   // n = nb de CANAUX ce tour

      vfloat32m1x2_t vt2 = __riscv_vlsseg2e32_v_f32m1x2(tl, input_increment, n);
      vfloat32m1_t vtl = __riscv_vget_v_f32m1x2_f32m1(vt2, 0);
      vfloat32m1_t vtr = __riscv_vget_v_f32m1x2_f32m1(vt2, 1);

      vfloat32m1x2_t vb2 = __riscv_vlsseg2e32_v_f32m1x2(bl, input_increment, n);
      vfloat32m1_t vbl = __riscv_vget_v_f32m1x2_f32m1(vb2, 0);
      vfloat32m1_t vbr = __riscv_vget_v_f32m1x2_f32m1(vb2, 1);

      vfloat32m1_t vt = __riscv_vfmacc_vf_f32m1(
          vtl, valphah, __riscv_vfsub_vv_f32m1(vtr, vtl, n), n);
      vfloat32m1_t vb = __riscv_vfmacc_vf_f32m1(
          vbl, valphah, __riscv_vfsub_vv_f32m1(vbr, vbl, n), n);

      vfloat32m1_t vo = __riscv_vfmacc_vf_f32m1(
          vt, valphav, __riscv_vfsub_vv_f32m1(vb, vt, n), n);

      __riscv_vsse32_v_f32m1(o, output_increment, vo, n);

      // Avance d'un bloc de n canaux.
      tl = (const float*) ((uintptr_t) tl + n * input_increment);
      bl = (const float*) ((uintptr_t) bl + n * input_increment);
      o  = (float*)       ((uintptr_t) o  + n * output_increment);
      c -= n;
    } while (c != 0);
  } while (--output_pixels != 0);
}
