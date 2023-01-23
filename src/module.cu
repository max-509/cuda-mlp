#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "perceptron/MultilayerPerceptron.h"

#include "perceptron/activations/IActivation.h"
#include "perceptron/activations/IdentityActivation.h"
#include "perceptron/activations/SigmoidActivation.h"
#include "perceptron/activations/ReLUActivation.h"

#include "perceptron/losses/ILoss.h"
#include "perceptron/losses/SquaredLoss.h"

#include "perceptron/optimizers/IOptimizer.h"
#include "perceptron/optimizers/SGD.h"

#include <memory>

#define REGISTER_ACTIVATION(CLASS_NAME)                                                                         \
py::class_<activations::CLASS_NAME, std::shared_ptr<activations::CLASS_NAME>,                                   \
activations::IActivation>(m, #CLASS_NAME)                                                                       \
.def(py::init<>())                                                                                              \
.def("compute", py::overload_cast<tensors::TensorReadOnly2D<float, false>>(&activations::IActivation::compute)) \
.def("compute", py::overload_cast<tensors::TensorReadOnly2D<float, true>>(&activations::IActivation::compute))  \
.def("compute",                                                                                                 \
py::overload_cast<tensors::TensorReadOnly2D<float, false>,                                                      \
tensors::TensorWriteable2D<float>>(&activations::IActivation::compute))                                         \
.def("compute",                                                                                                 \
py::overload_cast<tensors::TensorReadOnly2D<float, true>,                                                       \
tensors::TensorWriteable2D<float>>(&activations::IActivation::compute))                                         \
.def("derivative",                                                                                              \
py::overload_cast<tensors::TensorReadOnly2D<float, false>>(&activations::IActivation::derivative))              \
.def("derivative",                                                                                              \
py::overload_cast<tensors::TensorReadOnly2D<float, true>>(&activations::IActivation::derivative))               \
.def("derivative",                                                                                              \
py::overload_cast<tensors::TensorReadOnly2D<float, false>,                                                      \
tensors::TensorWriteable2D<float>>(&activations::IActivation::derivative))                                      \
.def("derivative",                                                                                              \
py::overload_cast<tensors::TensorReadOnly2D<float, true>,                                                       \
tensors::TensorWriteable2D<float>>(&activations::IActivation::derivative))

#define REGISTER_LOSS(CLASS_NAME)                                                                               \
py::class_<losses::CLASS_NAME, std::shared_ptr<losses::CLASS_NAME>, losses::ILoss>(m, #CLASS_NAME)              \
.def(py::init<>())                                                                                              \
.def("compute",                                                                                                 \
py::overload_cast<tensors::TensorReadOnly2D<float, false>,                                                      \
tensors::TensorReadOnly2D<float, false>>(&losses::ILoss::compute))                                              \
.def("compute",                                                                                                 \
py::overload_cast<tensors::TensorReadOnly2D<float, false>,                                                      \
tensors::TensorReadOnly2D<float, true>>(&losses::ILoss::compute))                                               \
.def("compute",                                                                                                 \
py::overload_cast<tensors::TensorReadOnly2D<float, true>,                                                       \
tensors::TensorReadOnly2D<float, false>>(&losses::ILoss::compute))                                              \
.def("compute",                                                                                                 \
py::overload_cast<tensors::TensorReadOnly2D<float, true>,                                                       \
tensors::TensorReadOnly2D<float, true>>(&losses::ILoss::compute))                                               \
.def("derivative",                                                                                              \
py::overload_cast<tensors::TensorReadOnly2D<float, false>,                                                      \
tensors::TensorReadOnly2D<float, false>>(&losses::ILoss::derivative))                                           \
.def("derivative",                                                                                              \
py::overload_cast<tensors::TensorReadOnly2D<float, false>,                                                      \
tensors::TensorReadOnly2D<float, true>>(&losses::ILoss::derivative))                                            \
.def("derivative",                                                                                              \
py::overload_cast<tensors::TensorReadOnly2D<float, true>,                                                       \
tensors::TensorReadOnly2D<float, false>>(&losses::ILoss::derivative))                                           \
.def("derivative",                                                                                              \
py::overload_cast<tensors::TensorReadOnly2D<float, true>,                                                       \
tensors::TensorReadOnly2D<float, true>>(&losses::ILoss::derivative))                                            \
.def("derivative",                                                                                              \
py::overload_cast<tensors::TensorReadOnly2D<float, false>,                                                      \
tensors::TensorReadOnly2D<float, false>,                                                                        \
tensors::TensorWriteable2D<float>>(&losses::ILoss::derivative))                                                 \
.def("derivative",                                                                                              \
py::overload_cast<tensors::TensorReadOnly2D<float, false>,                                                      \
tensors::TensorReadOnly2D<float, true>,                                                                         \
tensors::TensorWriteable2D<float>>(&losses::ILoss::derivative))                                                 \
.def("derivative",                                                                                              \
py::overload_cast<tensors::TensorReadOnly2D<float, true>,                                                       \
tensors::TensorReadOnly2D<float, false>,                                                                        \
tensors::TensorWriteable2D<float>>(&losses::ILoss::derivative))                                                 \
.def("derivative",                                                                                              \
py::overload_cast<tensors::TensorReadOnly2D<float, true>,                                                       \
tensors::TensorReadOnly2D<float, true>,                                                                         \
tensors::TensorWriteable2D<float>>(&losses::ILoss::derivative))

using namespace perceptron;
using namespace pybind11::literals;
namespace py = pybind11;

// TODO: Finish

PYBIND11_MODULE(cuda_mlp, m) {
  m.doc() = "Just another implementation of multilayer perceptron, but under CUDA architecture";

  /** Tensors **/

  py::module_::create_extension_module("tensors", "Functions for tensors creating", m)
    .def("")

  /** Tensors **/



  /** Activations **/

  py::class_<activations::IActivation, std::shared_ptr<activations::IActivation>>(m, "IActivation")
      .def("compute", py::overload_cast<tensors::TensorReadOnly2D<float, false>>(&activations::IActivation::compute))
      .def("compute", py::overload_cast<tensors::TensorReadOnly2D<float, true>>(&activations::IActivation::compute))
      .def("compute",
           py::overload_cast<tensors::TensorReadOnly2D<float, false>,
                             tensors::TensorWriteable2D<float>>(&activations::IActivation::compute))
      .def("compute",
           py::overload_cast<tensors::TensorReadOnly2D<float, true>,
                             tensors::TensorWriteable2D<float>>(&activations::IActivation::compute))
      .def("derivative",
           py::overload_cast<tensors::TensorReadOnly2D<float, false>>(&activations::IActivation::derivative))
      .def("derivative",
           py::overload_cast<tensors::TensorReadOnly2D<float, true>>(&activations::IActivation::derivative))
      .def("derivative",
           py::overload_cast<tensors::TensorReadOnly2D<float, false>,
                             tensors::TensorWriteable2D<float>>(&activations::IActivation::derivative))
      .def("derivative",
           py::overload_cast<tensors::TensorReadOnly2D<float, true>,
                             tensors::TensorWriteable2D<float>>(&activations::IActivation::derivative));

  REGISTER_ACTIVATION(IdentityActivation);
  REGISTER_ACTIVATION(ReLUActivation);
  REGISTER_ACTIVATION(SigmoidActivation);

  /** Activations **/



  /** Losses **/

  py::class_<losses::ILoss, std::shared_ptr<losses::ILoss>>(m, "ILoss")
      .def("compute",
           py::overload_cast<tensors::TensorReadOnly2D<float, false>,
                             tensors::TensorReadOnly2D<float, false>>(&losses::ILoss::compute))
      .def("compute",
           py::overload_cast<tensors::TensorReadOnly2D<float, false>,
                             tensors::TensorReadOnly2D<float, true>>(&losses::ILoss::compute))
      .def("compute",
           py::overload_cast<tensors::TensorReadOnly2D<float, true>,
                             tensors::TensorReadOnly2D<float, false>>(&losses::ILoss::compute))
      .def("compute",
           py::overload_cast<tensors::TensorReadOnly2D<float, true>,
                             tensors::TensorReadOnly2D<float, true>>(&losses::ILoss::compute))
      .def("derivative",
           py::overload_cast<tensors::TensorReadOnly2D<float, false>,
                             tensors::TensorReadOnly2D<float, false>>(&losses::ILoss::derivative))
      .def("derivative",
           py::overload_cast<tensors::TensorReadOnly2D<float, false>,
                             tensors::TensorReadOnly2D<float, true>>(&losses::ILoss::derivative))
      .def("derivative",
           py::overload_cast<tensors::TensorReadOnly2D<float, true>,
                             tensors::TensorReadOnly2D<float, false>>(&losses::ILoss::derivative))
      .def("derivative",
           py::overload_cast<tensors::TensorReadOnly2D<float, true>,
                             tensors::TensorReadOnly2D<float, true>>(&losses::ILoss::derivative))
      .def("derivative",
           py::overload_cast<tensors::TensorReadOnly2D<float, false>,
                             tensors::TensorReadOnly2D<float, false>,
                             tensors::TensorWriteable2D<float>>(&losses::ILoss::derivative))
      .def("derivative",
           py::overload_cast<tensors::TensorReadOnly2D<float, false>,
                             tensors::TensorReadOnly2D<float, true>,
                             tensors::TensorWriteable2D<float>>(&losses::ILoss::derivative))
      .def("derivative",
           py::overload_cast<tensors::TensorReadOnly2D<float, true>,
                             tensors::TensorReadOnly2D<float, false>,
                             tensors::TensorWriteable2D<float>>(&losses::ILoss::derivative))
      .def("derivative",
           py::overload_cast<tensors::TensorReadOnly2D<float, true>,
                             tensors::TensorReadOnly2D<float, true>,
                             tensors::TensorWriteable2D<float>>(&losses::ILoss::derivative));

  REGISTER_LOSS(SquaredLoss);

  /** Losses **/



  /** Optimizers **/

  py::class_<optimizers::SGD::describer_t>(m, "sgd_describer_t")
      .def(py::init([](double lr, double weights_decay, double momentum, double dampening, bool nesterov) {
             return optimizers::SGD::describer_t{lr, weights_decay, momentum, dampening, nesterov};
           }), py::kw_only(), "lr"_a = optimizers::SGD::DEFAULT_LR,
           "weights_decay"_a = optimizers::SGD::DEFAULT_WEIGHTS_DECAY, "momentum"_a = optimizers::SGD::DEFAULT_MOMENTUM,
           "dampening"_a = optimizers::SGD::DEFAULT_DAMPENING, "nesterov"_a = optimizers::SGD::DEFAULT_NESTEROV);

  /** Optimizers **/



  /** Layer **/

  py::class_<layer::layer_params_t>(m, "layer_params_t")
      .def(py::init([](size_type layer_size,
                       std::shared_ptr<activations::IActivation> activation) {
        return layer::layer_params_t{layer_size, activation};
      }), "layer_size"_a, "activation"_a)
      .def(py::init([](size_type layer_size,
                       std::shared_ptr<activations::IActivation> activation,
                       std::unordered_map<std::string, std::any> config) {
        return layer::layer_params_t{layer_size, activation, config};
      }), "layer_size"_a, "activation"_a, "config"_a)
      .def_readonly("layer_size", &layer::layer_params_t::layer_size)
      .def_readonly("activation", &layer::layer_params_t::activation)
      .def_readonly("config", &layer::layer_params_t::config);

  /** Layer **/



  /** MultilayerPerceptron **/

  py::class_<MultilayerPerceptron::MultilayerPerceptronBuilder,
             std::shared_ptr<MultilayerPerceptron::MultilayerPerceptronBuilder>>(m, "MultilayerPerceptronBuilder")
      .def(py::init([]() { return std::make_shared<MultilayerPerceptron::MultilayerPerceptronBuilder>(); }))
      .def("setBatchSize", &MultilayerPerceptron::MultilayerPerceptronBuilder::setBatchSize, "batch_size"_a)
      .def("setEtol", &MultilayerPerceptron::MultilayerPerceptronBuilder::setEtol, "etol"_a)
      .def("setNEpoch", &MultilayerPerceptron::MultilayerPerceptronBuilder::setNEpoch, "n_epoch"_a)
      .def("setMaxNotChangeIter",
           &MultilayerPerceptron::MultilayerPerceptronBuilder::setMaxNotChangeIter,
           "max_not_change_iter"_a)
      .def("setSeed", &MultilayerPerceptron::MultilayerPerceptronBuilder::setSeed, "seed"_a)
      .def("build",
           &MultilayerPerceptron::MultilayerPerceptronBuilder::build<optimizers::SGD::describer_t>,
           py::return_value_policy::take_ownership);

  // TODO: Make fit with py::tuple
//  py::class_<MultilayerPerceptron>(m, "MultilayerPerceptron")
//      .def("fit", py::overload_cast<features_labels_pair_t<false, false>>(&MultilayerPerceptron::fit))
//      .def("fit", py::overload_cast<features_labels_pair_t<false, true>>(&MultilayerPerceptron::fit))
//      .def("fit", py::overload_cast<features_labels_pair_t<true, false>>(&MultilayerPerceptron::fit))
//      .def("fit", py::overload_cast<features_labels_pair_t<true, true>>(&MultilayerPerceptron::fit));

  /** MultilayerPerceptron **/
}
