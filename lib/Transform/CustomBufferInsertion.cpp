//===- CustomBufferInsertion.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-stream/Transform/CustomBufferInsertion.h"
#include "circt-stream/Transform/Passes.h"
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace circt;
using namespace circt::handshake;
using namespace circt_stream;

namespace circt_stream {
#define GEN_PASS_DEF_CUSTOMBUFFERINSERTION
#include "circt-stream/Transform/Passes.h.inc"
} // namespace circt_stream

namespace {
class CustomBufferInsertionPass
    : public circt_stream::impl::CustomBufferInsertionBase<
          CustomBufferInsertionPass> {
public:
  void runOnOperation() override {}
};
} // namespace
