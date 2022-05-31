//===- StreamDialect.cpp - Stream dialect --------------........-*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-stream/Dialect/Stream/StreamDialect.h"

#include "circt-stream/Dialect/Stream/StreamOps.h"
#include "circt-stream/Dialect/Stream/StreamTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::stream;

#include "circt-stream/Dialect/Stream/StreamOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Stream dialect.
//===----------------------------------------------------------------------===//

void StreamDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt-stream/Dialect/Stream/StreamOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt-stream/Dialect/Stream/StreamOpsTypes.cpp.inc"
      >();
}

#define GET_TYPEDEF_CLASSES
#include "circt-stream/Dialect/Stream/StreamOpsTypes.cpp.inc"
