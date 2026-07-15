#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern "C" {
typedef void (*ukernel_fn)(size_t, size_t, const float**, size_t, const float*,
                           float*, size_t);
void xnn_f32_ibilinear_chw_ukernel__scalar_p2(size_t, size_t, const float**,
                                              size_t, const float*, float*,
                                              size_t);
void xnn_f32_ibilinear_chw_ukernel__rvv_u1v(size_t, size_t, const float**,
                                            size_t, const float*, float*,
                                            size_t);
void xnn_f32_ibilinear_chw_ukernel__rvv_u2v(size_t, size_t, const float**,
                                            size_t, const float*, float*,
                                            size_t);
void xnn_f32_ibilinear_chw_ukernel__rvv_stride_u2v(size_t, size_t,
                                                   const float**, size_t,
                                                   const float*, float*,
                                                   size_t);
void xnn_f32_ibilinear_chw_ukernel__rvv_u2v_off(size_t, size_t, const float**,
                                                size_t, const float*, float*,
                                                size_t);

void xnn_f32_ibilinear_chw_ukernel__rvv_u2v_inv(size_t, size_t, const float**,
                                                size_t, const float*, float*,
                                                size_t);

void xnn_f32_ibilinear_chw_ukernel__rvv_u1v_pack(size_t, size_t, const float**,
                                                 size_t, const float*, float*,
                                                 size_t);
void xnn_f32_ibilinear_chw_ukernel__rvv_u2v_pack(size_t, size_t, const float**,
                                                 size_t, const float*, float*,
                                                 size_t);
}

int main(int argc, char** argv) {
  std::string k = argc > 1 ? argv[1] : "u2v";
  ukernel_fn fn =
      k == "scalar"     ? xnn_f32_ibilinear_chw_ukernel__scalar_p2
      : k == "u1v"      ? xnn_f32_ibilinear_chw_ukernel__rvv_u1v
      : k == "stride"   ? xnn_f32_ibilinear_chw_ukernel__rvv_stride_u2v
      : k == "u2v_off"  ? xnn_f32_ibilinear_chw_ukernel__rvv_u2v_off
      : k == "u2v_inv"  ? xnn_f32_ibilinear_chw_ukernel__rvv_u2v_inv
      : k == "u1v_pack" ? xnn_f32_ibilinear_chw_ukernel__rvv_u1v_pack
      : k == "u2v_pack" ? xnn_f32_ibilinear_chw_ukernel__rvv_u2v_pack
                        : xnn_f32_ibilinear_chw_ukernel__rvv_u2v;

  const size_t channels = 32, output_pixels = 32 * 32;
  const size_t input_stride = 4 * output_pixels;
  const size_t iters = 20;

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> d;

  std::vector<float> input((channels - 1) * input_stride + 4 * output_pixels);
  std::vector<float> weights(output_pixels * 2);
  std::vector<float> output(output_pixels * channels);
  std::vector<const float*> indir(output_pixels * 2);

  for (auto& x : input) x = d(rng);
  for (auto& x : weights) x = d(rng);
  for (size_t i = 0; i < indir.size(); i++) indir[i] = input.data() + 2 * i;
  std::shuffle(indir.begin(), indir.end(), rng);

  for (size_t it = 0; it < iters; it++)
    fn(output_pixels, channels, indir.data(), 0, weights.data(), output.data(),
       input_stride * sizeof(float));

  volatile float sink = output[0];
  return (int)sink & 0;
}
