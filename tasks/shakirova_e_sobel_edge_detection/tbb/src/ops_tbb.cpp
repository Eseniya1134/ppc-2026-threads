#include "shakirova_e_sobel_edge_detection/tbb/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range2d.h>
#include <oneapi/tbb/parallel_for.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "shakirova_e_sobel_edge_detection/common/include/common.hpp"

namespace shakirova_e_sobel_edge_detection {

ShakirovaESobelEdgeDetectionTBB::ShakirovaESobelEdgeDetectionTBB(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool ShakirovaESobelEdgeDetectionTBB::ValidationImpl() {
  return GetInput().IsValid();
}

bool ShakirovaESobelEdgeDetectionTBB::PreProcessingImpl() {
  const auto &img = GetInput();
  GetOutput().assign(static_cast<size_t>(img.width) * static_cast<size_t>(img.height), 0);
  return true;
}

bool ShakirovaESobelEdgeDetectionTBB::RunImpl() {
  const auto &img = GetInput();
  const int h = img.height;
  const int w = img.width;
  const auto &inp = img.pixels;
  auto &out = GetOutput();

  constexpr std::array<std::array<int, 3>, 3> k_gx = {{{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}}};
  constexpr std::array<std::array<int, 3>, 3> k_gy = {{{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}}};

  tbb::parallel_for(tbb::blocked_range2d<int>(1, h - 1, 32, 1, w - 1, 64), [&](const tbb::blocked_range2d<int> &r) {
    for (int row = r.rows().begin(); row < r.rows().end(); ++row) {
      for (int col = r.cols().begin(); col < r.cols().end(); ++col) {
        int gx = 0;
        int gy = 0;
        for (int ky = -1; ky <= 1; ++ky) {
          for (int kx = -1; kx <= 1; ++kx) {
            const int pixel = inp[static_cast<size_t>((row + ky) * w) + static_cast<size_t>(col + kx)];
            const auto ky_idx = static_cast<size_t>(ky + 1);
            const auto kx_idx = static_cast<size_t>(kx + 1);
            gx += pixel * k_gx[ky_idx][kx_idx];
            gy += pixel * k_gy[ky_idx][kx_idx];
          }
        }
        const int magnitude = static_cast<int>(std::sqrt(static_cast<double>((gx * gx) + (gy * gy))));
        out[static_cast<size_t>(row * w) + static_cast<size_t>(col)] = std::clamp(magnitude, 0, 255);
      }
    }
  });

  return true;
}

bool ShakirovaESobelEdgeDetectionTBB::PostProcessingImpl() {
  return true;
}

}  // namespace shakirova_e_sobel_edge_detection
