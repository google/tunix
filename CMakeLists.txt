# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.19...3.30)
project(tunix LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# C++ sources are absent when only the build metadata is copied (e.g. the
# Dockerfile dependency-caching layer), in which case no extension is built.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tunix/rl/_packing_ext.cc")
  find_package(
    Python 3.12
    REQUIRED
    COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule
  )

  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT
    COMMAND_ERROR_IS_FATAL ANY
  )
  find_package(nanobind CONFIG REQUIRED)

  # Builds `tunix/<package>/<name>.cc` into the `tunix.<package>` module.
  function(tunix_add_nanobind_extension package name)
    nanobind_add_module(
      ${name}
      STABLE_ABI
      NB_STATIC
      tunix/${package}/${name}.cc
    )
    target_compile_features(${name} PRIVATE cxx_std_17)
    set_target_properties(
      ${name}
      PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tunix/${package}"
    )
    install(TARGETS ${name} LIBRARY DESTINATION tunix/${package})
  endfunction()

  tunix_add_nanobind_extension(rl _packing_ext)
  tunix_add_nanobind_extension(utils _trajectory_logger_ext)
endif()
