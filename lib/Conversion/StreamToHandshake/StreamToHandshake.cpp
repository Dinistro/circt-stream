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
#include "circt/Conversion/StandardToHandshake.h"
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace circt::handshake;
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

// Functionality to share state when lowering, see CIRCT's HandshakeLowering
class StreamLowering : public HandshakeLowering {
 public:
  explicit StreamLowering(Region &r) : HandshakeLowering(r) {}

  virtual LogicalResult setControlOnlyPath(
      ConversionPatternRewriter &rewriter) override {
    if (failed(HandshakeLowering::setControlOnlyPath(rewriter)))
      return failure();

    Block *entryBlock = &r.front();
    Value ctrl = getBlockEntryControl(entryBlock);

    // Replace original return ops with new returns with additional control
    // input
    for (auto yieldOp : llvm::make_early_inc_range(r.getOps<StreamYieldOp>())) {
      rewriter.setInsertionPoint(yieldOp);
      SmallVector<Value, 8> operands(yieldOp->getOperands());
      operands.push_back(ctrl);
      rewriter.replaceOpWithNewOp<handshake::ReturnOp>(yieldOp, operands);
    }

    return success();
  }

  virtual LogicalResult test(ConversionPatternRewriter &rewriter) {
    return success();
  }
};

struct FuncOpLowering : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::FuncOp op, OpAdaptor adapter,
      ConversionPatternRewriter &rewriter) const override {
    // type conversion
    TypeConverter *typeConverter = getTypeConverter();
    FunctionType oldFuncType = op.getFunctionType().cast<FunctionType>();

    TypeConverter::SignatureConversion sig(oldFuncType.getNumInputs());
    SmallVector<Type, 1> newResults;
    SmallVector<Type, 1> newArgs;
    if (failed(
            typeConverter->convertSignatureArgs(oldFuncType.getInputs(), sig)))
      return failure();

    // add the ctrl input
    sig.addInputs({rewriter.getNoneType()});
    if (failed(typeConverter->convertTypes(oldFuncType.getResults(),
                                           newResults)) ||
        failed(
            rewriter.convertRegionTypes(&op.getBody(), *typeConverter, &sig)))
      return failure();

    newResults.push_back(rewriter.getNoneType());
    auto newFuncType =
        rewriter.getFunctionType(sig.getConvertedTypes(), newResults);

    SmallVector<NamedAttribute, 4> attributes;
    for (const auto &attr : op->getAttrs()) {
      if (attr.getName() == SymbolTable::getSymbolAttrName() ||
          attr.getName() == function_interface_impl::getTypeAttrName())
        continue;
      attributes.push_back(attr);
    }

    auto newFuncOp = rewriter.create<handshake::FuncOp>(
        op.getLoc(), op.getName(), newFuncType, attributes);
    rewriter.inlineRegionBefore(op.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());

    rewriter.eraseOp(op);

    // TODO cannot use that here, due to changed block argument types
    // HandshakeLowering fol(newFuncOp.getBody());
    // if (failed(lowerRegion(fol, false, false))) return failure();
    newFuncOp.resolveArgAndResNames();

    return success();
  }
};

struct ReturnOpLowering : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::ReturnOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // TODO find a cleaner way to do that
    SmallVector<Value, 4> operands;
    for (Value v : adaptor.getOperands()) operands.push_back(v);

    Value ctrl = *(rewriter.getBlock()->args_rbegin());
    assert(ctrl.getType().isa<NoneType>());
    operands.push_back(ctrl);
    rewriter.replaceOpWithNewOp<handshake::ReturnOp>(op, operands);
    return success();
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
    StreamMapLowering,
    FuncOpLowering,
    ReturnOpLowering
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

    // Patterns to lower stream dialect operations
    populateStreamToHandshakePatterns(typeConverter, patterns);
    target.addLegalOp<ModuleOp>();
    target.addLegalDialect<handshake::HandshakeDialect>();
    target.addIllegalDialect<func::FuncDialect>();
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

