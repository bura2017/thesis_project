Branch-and-Cut Program version 1.0 2017

Program solves integer linear systems with Branch-and-Cut algorithm using GPU technologies.

INSTALL
-------
- Preliminary CUDA Toolkit must be installed. Program was compiled with nvcc and wasn't checked using another compilers.

- Compile with flag -lcublas

- Input file is prescribed in main-file, if it's not program generates satisfiable problem itself.

- Program has several variations of Dual Simplex algorithm. Released Branch-and-Cut algorithm uses "DualSimplexDevSync" in the beginning and "DualSimplexDouble" afterwards.

- Matrix multiplication is used in "BranchAndCut/initMatrix" function and counted in three ways: on cpu, on gpu and usin cublas library. Current version use the second way.

- Matrix transformation is the way how matrix is transformed in simplex method.

=========================================================================

Copyright (c) 2016 ISP RAS

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
