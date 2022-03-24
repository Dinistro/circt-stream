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

#include "mlir/Pass/Pass.h"

namespace mlir {

namespace scf {
class SCFDialect;
}

namespace arith {
class ArithmeticDialect;
}

namespace func {
class FuncDialect;
class FuncOp;
}  // namespace func

namespace standalone {

class StandaloneDialect;

// Generate the classes which represent the passes
#define GEN_PASS_CLASSES
#include "Standalone/Conversion/Passes.h.inc"

}  // namespace standalone
}  // namespace mlir

#endif  // CONVERSION_PASSDETAIL_H
