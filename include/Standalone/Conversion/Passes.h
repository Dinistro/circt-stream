//===- Passes.h - Conversion Pass Construction and Registration -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_CONVERSION_PASSES_H
#define STANDALONE_CONVERSION_PASSES_H

#include "Standalone/Conversion/StandaloneToScf.h"
#include "Standalone/Conversion/StreamToHandshake.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace standalone {
/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "Standalone/Conversion/Passes.h.inc"

}  // namespace standalone
}  // namespace mlir

#endif  // STANDALONE_CONVERSION_PASSES_H

