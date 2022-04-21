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
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace mlir;
using namespace standalone;
using namespace stream;

namespace {

class StreamTypeConverter : public TypeConverter {
 public:
  StreamTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](StreamType type) { return type.getElementType(); });
  }
};

struct StreamMapLowering : public OpConversionPattern<StreamMap> {
  using OpConversionPattern<StreamMap>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      StreamMap op, OpAdaptor adapter,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Block *condBlock = rewriter.getInsertionBlock();
    auto opPosition = rewriter.getInsertionPoint();
    Block *remainingOpsBlock = rewriter.splitBlock(condBlock, opPosition);

    ValueRange operands = op->getOperands();
    auto &region = op.getRegion();
    rewriter.setInsertionPointToEnd(condBlock);
    rewriter.create<cf::BranchOp>(loc, &region.front(), operands);

    for (Block &block : region) {
      if (auto terminator =
              dyn_cast<stream::StreamYieldOp>(block.getTerminator())) {
        ValueRange terminatorOperands = terminator->getOperands();
        rewriter.setInsertionPointToEnd(&block);
        rewriter.create<cf::BranchOp>(loc, remainingOpsBlock,
                                      terminatorOperands);
        rewriter.eraseOp(terminator);
      }
    }

    rewriter.inlineRegionBefore(region, remainingOpsBlock);

    SmallVector<Value> vals;
    SmallVector<Location> argLocs(op->getNumResults(), op->getLoc());
    for (auto arg :
         remainingOpsBlock->addArguments(op->getResultTypes(), argLocs))
      vals.push_back(arg);
    rewriter.replaceOp(op, vals);
    return success();
  }
};

static void populateStreamToHandshakePatterns(
    StreamTypeConverter &typeConverter, RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
    StreamMapLowering
  >(typeConverter, patterns.getContext());
  // clang-format on
}

class StreamToHandshakePass
    : public StreamToHandshakeBase<StreamToHandshakePass> {
 public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    StreamTypeConverter typeConverter;
    ConversionTarget target(getContext());

    // Converts the types used in operations of the func dialect
    populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns,
                                                             typeConverter);
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });

    populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](func::CallOp op) { return typeConverter.isLegal(op); });

    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addLegalOp<ModuleOp>();

    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
             isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                              typeConverter) ||
             isLegalForReturnOpTypeConversionPattern(op, typeConverter);
    });

    // Patterns to lower stream dialect operations
    populateStreamToHandshakePatterns(typeConverter, patterns);
    target.addIllegalDialect<StreamDialect>();

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};
}  // namespace

std::unique_ptr<Pass> mlir::stream::createStreamToHandshakePass() {
  return std::make_unique<StreamToHandshakePass>();
}

