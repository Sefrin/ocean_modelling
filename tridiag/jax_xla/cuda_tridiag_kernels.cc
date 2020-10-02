/* Copyright 2019 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "cuda_tridiag_kernels.h"

#include "kernel_pybind11_helpers.h"
#include "include/pybind11/pybind11.h"

namespace jax {
namespace {

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["cuda_tridiag_double"] = EncapsulateFunction(CudaTridiagDouble);
  dict["cuda_tridiag_float"] = EncapsulateFunction(CudaTridiagFloat);
  return dict;
}

PYBIND11_MODULE(cuda_tridiag_kernels, m) {
  m.def("registrations", &Registrations);
  m.def("cuda_tridiag_descriptor", [](std::int64_t total_size, std::int64_t num_systems, std::int64_t system_depth) {
      std::string result = BuildCudaTridiagDescriptor(total_size, num_systems, system_depth);
      return pybind11::bytes(result);
    });
}

}  // namespace
}  // namespace jax
