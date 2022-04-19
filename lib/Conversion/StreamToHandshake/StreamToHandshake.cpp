//===- StreamToHandshake.cpp - Translate Stream into Handshake ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This is the main Stream to Handshake Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "Standalone/Conversion/StreamToHandshake.h"

#include "../PassDetail.h"
#include "Standalone/Dialect/Stream/StreamDialect.h"
#include "Standalone/Dialect/Stream/StreamOps.h"
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace mlir;
using namespace standalone;
using namespace stream;

namespace {

struct StreamMapLowering : public OpRewritePattern<StreamMap> {
  using OpRewritePattern<StreamMap>::OpRewritePattern;

  LogicalResult matchAndRewrite(StreamMap negToZeroOp,
                                PatternRewriter &rewriter) const override {
    Location loc = negToZeroOp.getLoc();
    return failure();
  }
};

static void populateStreamToHandshakePatterns(RewritePatternSet &patterns) {
  patterns.add<StreamMapLowering>(patterns.getContext());
}

class StreamToHandshakePass
    : public StreamToHandshakeBase<StreamToHandshakePass> {
 public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateStreamToHandshakePatterns(patterns);

    ConversionTarget target(getContext());
    target.addIllegalOp<StreamMap>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
}  // namespace

std::unique_ptr<Pass> mlir::stream::createStreamToHandshakePass() {
  return std::make_unique<StreamToHandshakePass>();
}

