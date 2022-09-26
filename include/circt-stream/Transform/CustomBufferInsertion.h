//===- CustomBufferInsertion.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_STREAM_TRANSFORM_CUSTOMBUFFERINSERTION_H_
#define CIRCT_STREAM_TRANSFORM_CUSTOMBUFFERINSERTION_H_

#include <memory>

namespace mlir {
class Pass;
}

namespace circt_stream {
#define GEN_PASS_DECL_CUSTOMBUFFERINSERTION
#include "circt-stream/Transform/Passes.h.inc"
} // namespace circt_stream

#endif // CIRCT_STREAM_TRANSFORM_CUSTOMBUFFERINSERTION_H_
