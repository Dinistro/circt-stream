# Query to RTL

This project aims to produce RTL or HDL code from queries. It uses [MLIR](https://mlir.llvm.org/) and LLVM's incubator project [CIRCT](https://circt.llvm.org/) to do so. The project structure is based on MLIR's standalone template. 

## Setup

### Prerequisites

This setup assumes that you have built CIRCT according to the getting [started page](https://circt.llvm.org/getting_started/).

It should not be required to install LLVM/MLIR nor CIRCT. In case it doesn't work, try to add the `DCMAKE_INSTALL_PREFIX` flag to install the libraries to custom directories, such that they do not mess with existing installations.

### Building

```sh
mkdir build && cd build
cmake -G Ninja .. \
    -DLLVM_DIR=$LLVM_BUILD/lib/cmake/llvm \
    -DMLIR_DIR=$LLVM_BUILD/lib/cmake/mlir \
    -DCIRCT_DIR=$CIRCT_BUILD/lib/cmake/circt \
    -DLLVM_EXTERNAL_LIT=$LLVM_BUILD/bin/llvm-lit
ninja
```

### Testing

```sh
ninja check-standalone
```

### Build documentation

**NOTE**: This is currently broken

To build the documentation from the TableGen description of the dialect operations, run
```sh
ninja mlir-doc
```

## TODO
* Should we change the repo to have CIRCT as a submodule?
* Check if the build actually works without installing the dependencies

