//===- PassDetail.h - Conversion Pass class details -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef CONVERSION_PASSDETAIL_H
#define CONVERSION_PASSDETAIL_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

namespace scf {
class SCFDialect;
}
namespace cf {
class ControlFlowDialect;
}

namespace arith {
class ArithmeticDialect;
}

namespace func {
class FuncDialect;
class FuncOp;
}  // namespace func

namespace memref {
class MemRefDialect;
}
}  // namespace mlir

namespace circt {
namespace handshake {
class HandshakeDialect;
}
}  // namespace circt

namespace circt_stream {

namespace stream {
class StreamDialect;
}

// Generate the classes which represent the passes
#define GEN_PASS_CLASSES
#include "circt-stream/Conversion/Passes.h.inc"

}  // namespace circt_stream

#endif  // CONVERSION_PASSDETAIL_H

