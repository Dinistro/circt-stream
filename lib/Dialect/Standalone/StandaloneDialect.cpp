//===- StandaloneDialect.cpp - Standalone dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/Standalone/StandaloneDialect.h"

#include "Standalone/Dialect/Standalone/StandaloneOps.h"

using namespace mlir;
using namespace mlir::standalone;

#include "Standalone/Dialect/Standalone/StandaloneOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Standalone dialect.
//===----------------------------------------------------------------------===//

void StandaloneDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Standalone/Dialect/Standalone/StandaloneOps.cpp.inc"
      >();
}
