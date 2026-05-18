#include "shakirova_e_sobel_edge_detection/all/include/ops_all.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

#include "shakirova_e_sobel_edge_detection/common/include/common.hpp"
#include "util/include/util.hpp"

namespace shakirova_e_sobel_edge_detection {

ShakirovaESobelEdgeDetectionALL::ShakirovaESobelEdgeDetectionALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool ShakirovaESobelEdgeDetectionALL::ValidationImpl() {
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized != 0) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
      return GetInput().IsValid();
    }
    return true;
  }
  return GetInput().IsValid();
}

bool ShakirovaESobelEdgeDetectionALL::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int w = 0;
  int h = 0;
  if (rank == 0) {
    w = GetInput().width;
    h = GetInput().height;
  }
  MPI_Bcast(&w, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&h, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput().assign(static_cast<size_t>(w) * static_cast<size_t>(h), 0);
  return true;
}

bool ShakirovaESobelEdgeDetectionALL::RunImpl() {
  int rank = 0;
  int mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  int w = 0;
  int h = 0;
  if (rank == 0) {
    w = GetInput().width;
    h = GetInput().height;
  }
  MPI_Bcast(&w, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&h, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (h < 3 || w < 3) {
    return true;
  }

  const int inner_rows = h - 2;
  const int num_stl = ppc::util::GetNumThreads();

  std::vector<int> row_counts(mpi_size);
  std::vector<int> row_offsets(mpi_size);
  for (int i = 0; i < mpi_size; ++i) {
    row_counts[i] = inner_rows / mpi_size + (i < inner_rows % mpi_size ? 1 : 0);
    row_offsets[i] = (i == 0) ? 0 : row_offsets[i - 1] + row_counts[i - 1];
  }

  const int local_inner = row_counts[rank];

  std::vector<int> send_counts(mpi_size);
  std::vector<int> send_displs(mpi_size);
  for (int i = 0; i < mpi_size; ++i) {
    if (row_counts[i] == 0) {
      send_counts[i] = 0;
      send_displs[i] = (i == 0) ? 0 : send_displs[i - 1];
    } else {
      send_counts[i] = (row_counts[i] + 2) * w;
      send_displs[i] = row_offsets[i] * w;
    }
  }

  const int local_buf_rows = (local_inner > 0) ? (local_inner + 2) : 0;
  std::vector<int> local_buf(static_cast<size_t>(local_buf_rows) * static_cast<size_t>(w), 0);

  const int *inp_ptr = (rank == 0) ? GetInput().pixels.data() : nullptr;
  MPI_Scatterv(inp_ptr, send_counts.data(), send_displs.data(), MPI_INT,
               local_buf.data(), local_buf_rows * w, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> local_out(static_cast<size_t>(local_inner) * static_cast<size_t>(w), 0);

  if (local_inner > 0) {
    const int *lb = local_buf.data();
    int *lo = local_out.data();

    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(num_stl));

    for (int tid = 0; tid < num_stl; ++tid) {
      threads.emplace_back([lb, lo, w, local_inner, num_stl, tid]() {
        const int chunk = local_inner / num_stl;
        const int row_begin = tid * chunk;
        const int row_end = (tid == num_stl - 1) ? local_inner : row_begin + chunk;

        for (int row = row_begin; row < row_end; ++row) {
          const int *prev = lb + static_cast<ptrdiff_t>((row) * w);
          const int *curr = lb + static_cast<ptrdiff_t>((row + 1) * w);
          const int *next = lb + static_cast<ptrdiff_t>((row + 2) * w);
          int *out_row    = lo + static_cast<ptrdiff_t>(row * w);

          for (int col = 1; col < w - 1; ++col) {
            const int gx = -prev[col - 1] + prev[col + 1]
                           - (2 * curr[col - 1]) + (2 * curr[col + 1])
                           - next[col - 1] + next[col + 1];
            const int gy = -prev[col - 1] - (2 * prev[col]) - prev[col + 1]
                           + next[col - 1] + (2 * next[col]) + next[col + 1];

            const int agx = std::abs(gx);
            const int agy = std::abs(gy);
            const int magnitude =
                (((std::max(agx, agy) * 123) + (std::min(agx, agy) * 51)) >> 7);
            out_row[col] = magnitude > 255 ? 255 : magnitude;
          }
        }
      });
    }

    for (auto &th : threads) {
      th.join();
    }
  }

  std::vector<int> recv_counts(mpi_size);
  std::vector<int> recv_displs(mpi_size);
  for (int i = 0; i < mpi_size; ++i) {
    recv_counts[i] = row_counts[i] * w;
    recv_displs[i] = (row_offsets[i] + 1) * w;
  }

  MPI_Gatherv(local_out.data(), local_inner * w, MPI_INT,
              GetOutput().data(), recv_counts.data(), recv_displs.data(), MPI_INT,
              0, MPI_COMM_WORLD);

  MPI_Bcast(GetOutput().data(), static_cast<int>(GetOutput().size()), MPI_INT,
            0, MPI_COMM_WORLD);

  return true;
}

bool ShakirovaESobelEdgeDetectionALL::PostProcessingImpl() {
  return true;
}

}  // namespace shakirova_e_sobel_edge_detection 
