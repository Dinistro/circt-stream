//===- StreamToHandshake.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares passes which together will lower the Stream dialect to
// the Handshake dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_STREAM_CONVERSION_STREAMTOHANDSHAKE_H_
#define CIRCT_STREAM_CONVERSION_STREAMTOHANDSHAKE_H_

#include <memory>

namespace mlir {
class Pass;
}

namespace circt_stream {
std::unique_ptr<mlir::Pass> createStreamToHandshakePass();
}
#endif // CIRCT_STREAM_CONVERSION_STREAMTOHANDSHAKE_H_
