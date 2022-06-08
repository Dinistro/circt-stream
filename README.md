# CIRCT Stream

This project aims to add a streaming abstraction on top of [CIRCT](https://circt.llvm.org/) which in itself depends on [MLIR](https://mlir.llvm.org/).

## Setup

### Prerequisites

This setup assumes that you have built CIRCT according to the getting [started page](https://circt.llvm.org/getting_started/).

It should not be required to install LLVM/MLIR nor CIRCT.

### Building

```sh
mkdir build && cd build
cmake -G Ninja .. \
    -DLLVM_DIR=$LLVM_BUILD/lib/cmake/llvm \
    -DMLIR_DIR=$LLVM_BUILD/lib/cmake/mlir \
    -DCIRCT_DIR=$CIRCT_BUILD/lib/cmake/circt \
    -DLLVM_EXTERNAL_LIT=$LLVM_BUILD/bin/llvm-lit \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG
ninja
```

### Testing

There are two types of tests. FileCheck tests can be executed with the following command:

```sh
ninja check-stream
```

To execute the integration tests an installation of Xilinx Vivado is required. Either provide the path to the vivado executable with the `-DVIVADO_PATH` flag to cmake or add it to the path.

```sh
ninja check-stream-integration
```
