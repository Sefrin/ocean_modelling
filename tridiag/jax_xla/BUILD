# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# JAX is Autograd and XLA

load("@org_tensorflow//tensorflow:tensorflow.bzl", "pybind_extension")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "kernel_pybind11_helpers",
    hdrs = ["kernel_pybind11_helpers.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":kernel_helpers",
        "@pybind11",
    ],
)

cc_library(
    name = "kernel_helpers",
    hdrs = ["kernel_helpers.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/base",
    ],
)

cc_library(
    name = "gpu_kernel_helpers",
    srcs = ["gpu_kernel_helpers.cc"],
    hdrs = ["gpu_kernel_helpers.h"],
    copts = [
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)


py_library(
    name = "tridiag",
    srcs = [
        "tridiag.py"
    ],
    deps = [
        ":gpu_support"
    ],
)

py_library(
    name = "gpu_support",
    deps = [
        ":cuda_tridiag_kernels"
    ],
)

cuda_library(
    name = "cuda_tridiag_kernels_lib",
    srcs = ["cuda_tridiag_kernels.cu.cc"],
    hdrs = ["cuda_tridiag_kernels.h"],
    deps = [
        ":gpu_kernel_helpers",
        ":kernel_helpers",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

pybind_extension(
    name = "cuda_tridiag_kernels",
    srcs = ["cuda_tridiag_kernels.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "cuda_keksetridiag_kernels",
    deps = [
        ":cuda_tridiag_kernels_lib",
        ":kernel_pybind11_helpers",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cudart_stub",
        "@local_config_cuda//cuda:cuda_headers",
        "@pybind11",
    ],
)