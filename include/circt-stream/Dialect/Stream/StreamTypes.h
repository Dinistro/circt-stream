//===- StreamTypes.h - Stream Types -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_STREAM_DIALECT_STREAM_TYPES_H
#define CIRCT_STREAM_DIALECT_STREAM_TYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "circt-stream/Dialect/Stream/StreamOpsTypes.h.inc"

#endif // CIRCT_STREAM_DIALECT_STREAM_TYPES_H
