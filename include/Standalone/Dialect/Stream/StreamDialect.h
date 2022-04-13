//===- StreamDialect.h - Stream dialect -------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_STREAMDIALECT_H
#define STANDALONE_STREAMDIALECT_H

#include "mlir/IR/Dialect.h"
// Do not remove, otherwise includes will be reorder and this breaks everything
#include "Standalone/Dialect/Stream/StreamOpsDialect.h.inc"

#endif  // STANDALONE_STREAMDIALECT_H
