//===- StreamDialect.cpp - Stream dialect --------------........-*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/Stream/StreamDialect.h"

#include "Standalone/Dialect/Stream/StreamOps.h"
#include "Standalone/Dialect/Stream/StreamTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::stream;

#include "Standalone/Dialect/Stream/StreamOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Stream dialect.
//===----------------------------------------------------------------------===//

void StreamDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Standalone/Dialect/Stream/StreamOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Standalone/Dialect/Stream/StreamOpsTypes.cpp.inc"
      >();
}

#define GET_TYPEDEF_CLASSES
#include "Standalone/Dialect/Stream/StreamOpsTypes.cpp.inc"
