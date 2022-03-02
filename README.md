# Query to RTL

This project aims to produce RTL or HDL code from queries. It uses [MLIR](https://mlir.llvm.org/) and LLVM's incubator project [CIRCT](https://circt.llvm.org/) to do so. The project structure is based on MLIR's standalone template. 

## Building

This setup assumes that you have built CIRCT according to the getting [started page](https://circt.llvm.org/getting_started/). 

**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with CMake in order to install `FileCheck` to the chosen installation prefix.

```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$LLVM_BUILD/lib/cmake/mlir -DCIRCT_DIR=$CIRCT_BUILD/lib/cmake/circt -DLLVM_EXTERNAL_LIT=$LLVM_BUILD/bin/llvm-lit 

```

To build the documentation from the TableGen description of the dialect operations, run
```sh
cmake --build . --target mlir-doc
```

## TODO
* Is there a way to skip the installing of the dependencies? Somehow MLIR\_DIR and CIRCT\_DIR only work when the projects were installed.
* Should we change the repo to have CIRCT as a submodule?

