//===- StandaloneToScf.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares passes which together will lower the Standalone dialect to
// the SCF dialect.
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_CONVERSION_STANDALONETOSCF_H_
#define STANDALONE_CONVERSION_STANDALONETOSCF_H_

#include <memory>

namespace mlir {
class Pass;

namespace standalone {
std::unique_ptr<mlir::Pass> createStandaloneToScfPass();
}  // namespace standalone

}  // namespace mlir
#endif  // STANDALONE_CONVERSION_STANDALONETOSCF_H_

