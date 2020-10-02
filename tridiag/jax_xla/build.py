#!/usr/bin/python
#
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
#
# Helper script for building JAX's libjax easily.


import argparse
import collections
import hashlib
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import urllib

# pylint: disable=g-import-not-at-top
if hasattr(urllib, "urlretrieve"):
  urlretrieve = urllib.urlretrieve
else:
  import urllib.request
  urlretrieve = urllib.request.urlretrieve

if hasattr(shutil, "which"):
  which = shutil.which
else:
  from distutils.spawn import find_executable as which
# pylint: enable=g-import-not-at-top


def shell(cmd):
  output = subprocess.check_output(cmd)
  return output.decode("UTF-8").strip()


# Python

def get_python_bin_path(python_bin_path_flag):
  """Returns the path to the Python interpreter to use."""
  return python_bin_path_flag or sys.executable


def get_python_version(python_bin_path):
  version_output = shell(
    [python_bin_path, "-c",
     "import sys; print(\"{}.{}\".format(sys.version_info[0], "
     "sys.version_info[1]))"])
  major, minor = map(int, version_output.split("."))
  return major, minor

def check_python_version(python_version):
  if python_version < (3, 6):
    print("JAX requires Python 3.6 or newer.")
    sys.exit(-1)


# Bazel

BAZEL_BASE_URI = "https://github.com/bazelbuild/bazel/releases/download/2.0.0/"
BazelPackage = collections.namedtuple("BazelPackage", ["file", "sha256"])
bazel_packages = {
    "Linux":
        BazelPackage(
            file="bazel-2.0.0-linux-x86_64",
            sha256=
            "4df79462c6c3ecdeeee7af99fc269b52ab1aa4828ef3bc359c1837d3fafeeee7"),
    "Darwin":
        BazelPackage(
            file="bazel-2.0.0-darwin-x86_64",
            sha256=
            "3eca4c96cfda97a9d5f8d3d0dec4155a5cc5ff339b10d3f35213c398bf13881e"),
}

def get_bazel_path(bazel_path_flag):
  """Returns the path to a Bazel binary, downloading Bazel if not found."""
  if bazel_path_flag:
    return bazel_path_flag

  bazel = which("bazel")
  if bazel:
    return bazel

  print("Cannot find or download bazel. Please install bazel.")
  sys.exit(-1)


def check_bazel_version(bazel_path, min_version, max_version):
  """Checks Bazel's version is in the range [`min_version`, `max_version`)."""
  version_output = shell([bazel_path, "--bazelrc=/dev/null", "version"])
  match = re.search("Build label: *([0-9\\.]+)[^0-9\\.]", version_output)
  if match is None:
    print("Warning: bazel installation is not a release version. Make sure "
          "bazel is at least {}".format(min_version))
    return
  version = match.group(1)
  min_ints = [int(x) for x in min_version.split(".")]
  actual_ints = [int(x) for x in match.group(1).split(".")]
  if min_ints > actual_ints:
    print("Outdated bazel revision (>= {} required, found {})".format(
        min_version, version))
    sys.exit(-1)
  if max_version is not None:
    max_ints = [int(x) for x in max_version.split(".")]
    if actual_ints >= max_ints:
      print("Please downgrade your bazel revision to build JAX (>= {} and < {}"
            " required, found {})".format(min_version, max_version, version))
      sys.exit(-1)


BAZELRC_TEMPLATE = """
# Flag to enable remote config
common --experimental_repo_remote_exec

build --repo_env PYTHON_BIN_PATH="{python_bin_path}"
build --python_path="{python_bin_path}"
build --repo_env TF_NEED_CUDA="{tf_need_cuda}"
build --action_env TF_CUDA_COMPUTE_CAPABILITIES="{cuda_compute_capabilities}"
build --distinct_host_configuration=false
build --copt=-Wno-sign-compare
build -c opt
build:opt --copt=-march=native
build:opt --host_copt=-march=native
build:mkl_open_source_only --define=tensorflow_mkldnn_contraction_kernel=1

# Sets the default Apple platform to macOS.
build --apple_platform_type=macos
build --macos_minimum_os=10.9

# Make Bazel print out all options from rc files.
build --announce_rc

build --define open_source_build=true

# Disable enabled-by-default TensorFlow features that we don't care about.
build --define=no_aws_support=true
build --define=no_gcp_support=true
build --define=no_hdfs_support=true
build --define=no_kafka_support=true
build --define=no_ignite_support=true
build --define=grpc_no_ares=true

build:cuda --crosstool_top=@local_config_cuda//crosstool:toolchain
build:cuda --define=using_cuda=true --define=using_cuda_nvcc=true

build --spawn_strategy=standalone
build --strategy=Genrule=standalone

build --cxxopt=-std=c++14
build --host_cxxopt=-std=c++14

# Suppress all warning messages.
build:short_logs --output_filter=DONT_MATCH_ANYTHING
"""



def write_bazelrc(cuda_toolkit_path=None, cudnn_install_path=None, **kwargs):
  with open(".bazelrc", "w") as f:
    f.write(BAZELRC_TEMPLATE.format(**kwargs))
    if cuda_toolkit_path:
      f.write("build --action_env CUDA_TOOLKIT_PATH=\"{cuda_toolkit_path}\"\n"
              .format(cuda_toolkit_path=cuda_toolkit_path))
    if cudnn_install_path:
      f.write("build --action_env CUDNN_INSTALL_PATH=\"{cudnn_install_path}\"\n"
              .format(cudnn_install_path=cudnn_install_path))


def _parse_string_as_bool(s):
  """Parses a string as a boolean argument."""
  lower = s.lower()
  if lower == "true":
    return True
  elif lower == "false":
    return False
  else:
    raise ValueError("Expected either 'true' or 'false'; got {}".format(s))


def add_boolean_argument(parser, name, default=False, help_str=None):
  """Creates a boolean flag."""
  group = parser.add_mutually_exclusive_group()
  group.add_argument(
      "--" + name,
      nargs="?",
      default=default,
      const=True,
      type=_parse_string_as_bool,
      help=help_str)
  group.add_argument("--no" + name, dest=name, action="store_false")


def main():
  parser = argparse.ArgumentParser(
      description="Builds libjax from source.")
  parser.add_argument(
      "--bazel_path",
      help="Path to the Bazel binary to use. The default is to find bazel via "
      "the PATH; if none is found, downloads a fresh copy of bazel from "
      "GitHub.")
  parser.add_argument(
      "--python_bin_path",
      help="Path to Python binary to use. The default is the Python "
      "interpreter used to run the build script.")
  add_boolean_argument(
      parser,
      "enable_march_native",
      default=False,
      help_str="Generate code targeted to the current machine? This may "
          "increase performance, but may generate code that does not run on "
          "older machines.")
  add_boolean_argument(
      parser,
      "enable_mkl_dnn",
      default=True,
      help_str="Should we build with MKL-DNN enabled?")
  parser.add_argument(
      "--cuda_path",
      default=None,
      help="Path to the CUDA toolkit.")
  parser.add_argument(
      "--cudnn_path",
      default=None,
      help="Path to CUDNN libraries.")
  parser.add_argument(
      "--cuda_compute_capabilities",
      default="3.5,5.2,6.0,6.1,7.0",
      help="A comma-separated list of CUDA compute capabilities to support.")
  parser.add_argument(
      "--bazel_startup_options",
      action="append", default=[],
      help="Additional startup options to pass to bazel.")
  parser.add_argument(
      "--bazel_options",
      action="append", default=[],
      help="Additional options to pass to bazel.")
  args = parser.parse_args()


  os.chdir(os.path.dirname(__file__ or args.prog) or '.')

  # Find a working Bazel.
  bazel_path = get_bazel_path(args.bazel_path)
  check_bazel_version(bazel_path, min_version="2.0.0", max_version=None)
  print("Bazel binary path: {}".format(bazel_path))

  python_bin_path = get_python_bin_path(args.python_bin_path)
  print("Python binary path: {}".format(python_bin_path))
  python_version = get_python_version(python_bin_path)
  print("Python version: {}".format(".".join(map(str, python_version))))
  check_python_version(python_version)

  print("MKL-DNN enabled: {}".format("yes" if args.enable_mkl_dnn else "no"))
  print("-march=native: {}".format("yes" if args.enable_march_native else "no"))

  cuda_toolkit_path = args.cuda_path
  cudnn_install_path = args.cudnn_path

  if cuda_toolkit_path:
    print("CUDA toolkit path: {}".format(cuda_toolkit_path))
  if cudnn_install_path:
    print("CUDNN library path: {}".format(cudnn_install_path))
  print("CUDA compute capabilities: {}".format(args.cuda_compute_capabilities))
  write_bazelrc(
      python_bin_path=python_bin_path,
      tf_need_cuda=1,
      cuda_toolkit_path=cuda_toolkit_path,
      cudnn_install_path=cudnn_install_path,
      cuda_compute_capabilities=args.cuda_compute_capabilities)

  print("\nBuilding XLA and installing it in the jaxlib source tree...")
  config_args = args.bazel_options
  config_args += ["--config=short_logs"]
  if args.enable_march_native:
    config_args += ["--config=opt"]
  if args.enable_mkl_dnn:
    config_args += ["--config=mkl_open_source_only"]
  config_args += ["--config=cuda"]
  config_args += ["--define=xla_python_enable_gpu=true"]
  command = ([bazel_path] + args.bazel_startup_options +
    ["build", "tridiag", "--verbose_failures=true"])
  print(" ".join(command))
  shell(command)
  shutil.copy("bazel-bin/cuda_tridiag_kernels.so", ".")
#   shell([bazel_path, "shutdown"])


if __name__ == "__main__":
  main()
