#ifndef MLP_DEMO_RESULTS_SAVER_H
#define MLP_DEMO_RESULTS_SAVER_H

#include <csv.hpp>

#include "perceptron/tensors/Tensor2D.h"

#include <fstream>
#include <vector>
#include <string>

template<typename T, bool trans>
void
tensor_to_csv(perceptron::tensors::TensorReadOnly2D<T, trans> tensor,
              std::string path,
              std::vector<std::string> columns) {
  using perceptron::size_type;

  auto csv_file = std::ofstream{path};
  auto writer = csv::make_csv_writer(csv_file);
  auto nrows = tensor.get_nrows();
  auto ncols = tensor.get_ncols();

  auto tensor_row_buffer = std::vector<std::string>(ncols);

  writer << columns;
  for (size_type row = 0; row < nrows; ++row) {
    for (size_type col = 0; col < ncols; ++col) {
      tensor_row_buffer[col] = std::to_string(tensor(row, col));
    }
    writer << tensor_row_buffer;
  }
  writer.flush();
}

#endif //MLP_DEMO_RESULTS_SAVER_H
