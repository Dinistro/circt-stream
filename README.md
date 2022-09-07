# CIRCT Stream

This project aims to add a streaming abstraction on top of [CIRCT](https://circt.llvm.org/) which in itself depends on [MLIR](https://mlir.llvm.org/).

## Setup

### Prerequisites

Apart from `cmake` and `Ninja`, this project does not come with any prerequisites.
To reduce the amount of memory required, it is beneficial to install llvm's `lld`, as it is much more memory efficent than `ld`.

### Cloning

This project has a pinned version of CIRCT which in turn comes with a pinned verison of LLVM.
These submodules can directly be initialized when the repository is cloned:

```sh
git clone --recurse-submodules git@github.com:Dinistro/circt-stream.git
```

### Building

Before the `circt-stream` project can be built, its dependencies need to be built.

To make things easier, one should build CIRCT together with LLVM as follows:

```sh
mkdir circt/build && cd circt/build
cmake -G Ninja ../llvm/llvm \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_EXTERNAL_PROJECTS=circt \
    -DLLVM_EXTERNAL_CIRCT_SOURCE_DIR=.. \
    -DLLVM_USE_LINKER=lld \ # remove if lld isn't present
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_CCACHE_BUILD=1 # remove if CCache isn't present
ninja
```

After CIRCT has been built, one can build this project as follows: 

```sh
mkdir build && cd build
cmake -G Ninja .. \
    -DLLVM_DIR=$PWD/../circt/build/lib/cmake/llvm \
    -DMLIR_DIR=$PWD/../circt/build/lib/cmake/mlir \
    -DCIRCT_DIR=$PWD/../circt/build/lib/cmake/circt \
    -DLLVM_EXTERNAL_LIT=$PWD/../circt/build/bin/llvm-lit \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DLLVM_USE_LINKER=lld \ # remove if lld isn't present
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++
ninja
```

### Testing

There are two types of tests. FileCheck tests can be executed with the following command:

```sh
ninja check-stream
```

To execute the integration tests, an installation of Icarus Verilog and pythons `cocotb` and `cocotb-test` packages are required.

```sh
ninja check-stream-integration
```
