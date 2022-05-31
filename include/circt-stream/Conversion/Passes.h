//===- Passes.h - Conversion Pass Construction and Registration -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_STREAM_CONVERSION_PASSES_H
#define CIRCT_STREAM_CONVERSION_PASSES_H

#include "circt-stream/Conversion/StreamToHandshake.h"
#include "mlir/Pass/PassRegistry.h"

namespace circt_stream {
/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "circt-stream/Conversion/Passes.h.inc"

}  // namespace circt_stream

#endif  // CIRCT_STREAM_CONVERSION_PASSES_H

