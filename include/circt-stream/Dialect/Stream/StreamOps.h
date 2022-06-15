//===- StreamOps.h - Stream dialect ops -------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_STREAM_DIALECT_STREAM_OPS_H
#define CIRCT_STREAM_DIALECT_STREAM_OPS_H

#include "circt-stream/Dialect/Stream/StreamTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "circt-stream/Dialect/Stream/StreamOps.h.inc"

#endif // CIRCT_STREAM_DIALECT_STREAM_OPS_H
