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

  nanobind_add_module(
    _packing_ext
    STABLE_ABI
    NB_STATIC
    tunix/rl/_packing_ext.cc
  )
  target_compile_features(_packing_ext PRIVATE cxx_std_17)
  set_target_properties(
    _packing_ext
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tunix/rl"
  )
  install(TARGETS _packing_ext LIBRARY DESTINATION tunix/rl)
endif()
