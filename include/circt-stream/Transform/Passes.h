//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for CIRCT Stream transformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SREAM_TRANSFORMS_PASSES_H
#define CIRCT_SREAM_TRANSFORMS_PASSES_H

#include "circt-stream/Transform/CustomBufferInsertion.h"
#include "mlir/Pass/Pass.h"

namespace circt_stream {

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt-stream/Transform/Passes.h.inc"

} // namespace circt_stream

#endif // CIRCT_SREAM_TRANSFORMS_PASSES_H
