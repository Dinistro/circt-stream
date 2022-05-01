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

  LogicalResult test(ConversionPatternRewriter &rewriter) { return success(); }
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

// Assumes that the op producing the input date also produces a ctrl signal
// This assumption is invalid
template <typename Adaptor>
static Value getCtrlSignal(Adaptor adaptor) {
  assert(adaptor.getOperands().size() > 0);
  Value fstOp = adaptor.getOperands().front();
  Value ctrl;
  if (BlockArgument arg = fstOp.dyn_cast_or_null<BlockArgument>()) {
    Block *block = arg.getOwner();
    ctrl = block->getArguments().back();
  } else {
    // TODO only check for instances?
    Operation *defOp = fstOp.getDefiningOp();
    assert(dyn_cast<handshake::InstanceOp>(defOp) &&
           "can only deduce ctrl signal from InstanceOps");
    ctrl = defOp->getResults().back();
  }

  assert(ctrl.getType().isa<NoneType>());

  return ctrl;
}

struct ReturnOpLowering : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::ReturnOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value, 4> operands = llvm::to_vector<4>(adaptor.getOperands());

    Value ctrl = getCtrlSignal(adaptor);
    operands.push_back(ctrl);
    rewriter.replaceOpWithNewOp<handshake::ReturnOp>(op, operands);
    return success();
  }
};

/// Returns a name resulting from an operation, without discriminating
/// type information.
static std::string getBareOpName(Operation *op) {
  std::string name = op->getName().getStringRef().str();
  std::replace(name.begin(), name.end(), '.', '_');
  return name;
}

static std::string getFuncName(Operation *op) {
  std::string opName = getBareOpName(op);
  // TODO add unique id or something
  return opName;
}

Block *getTopLevelBock(Operation *op) {
  return &op->getParentOfType<ModuleOp>().getRegion().front();
}

// Builds a handshake::FuncOp and that represents the mapping funtion. This
// function is then instantiated and connected to its inputs and outputs.
struct StreamMapLowering : public OpConversionPattern<StreamMap> {
  using OpConversionPattern<StreamMap>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      StreamMap op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    TypeConverter *typeConverter = getTypeConverter();

    StreamLowering sl(op.getRegion());

    if (failed(lowerRegion<StreamYieldOp>(sl, false, false))) return failure();

    SmallVector<mlir::Type, 8> argTypes = {
        typeConverter->convertType(op.input().getType())};

    SmallVector<mlir::Type, 8> resTypes = {
        typeConverter->convertType(op.res().getType())};

    auto noneType = rewriter.getNoneType();
    argTypes.push_back(noneType);
    resTypes.push_back(noneType);

    auto func_type = rewriter.getFunctionType(argTypes, resTypes);
    rewriter.setInsertionPointToStart(getTopLevelBock(op));
    FuncOp newFuncOp = rewriter.create<FuncOp>(rewriter.getUnknownLoc(),
                                               getFuncName(op), func_type);
    // Makes function private
    SymbolTable::setSymbolVisibility(newFuncOp,
                                     SymbolTable::Visibility::Private);
    rewriter.inlineRegionBefore(op.getRegion(), newFuncOp.getBody(),
                                newFuncOp.end());
    newFuncOp.resolveArgAndResNames();

    SmallVector<Value, 2> operands;
    for (auto operand : adaptor.getOperands()) operands.push_back(operand);

    operands.push_back(getCtrlSignal(adaptor));

    rewriter.setInsertionPoint(op);
    InstanceOp instance = rewriter.create<InstanceOp>(loc, newFuncOp, operands);

    // The ctrl signal has to be ignored
    rewriter.replaceOp(op, instance.getResults().drop_back());

    return success();
  }
};

static void populateStreamToHandshakePatterns(
    StreamTypeConverter &typeConverter, RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
    FuncOpLowering,
    ReturnOpLowering,
    StreamMapLowering//,
    //StreamFilterLowering
  >(typeConverter, patterns.getContext());
  // clang-format on
}

class StreamToHandshakePass
    : public StreamToHandshakeBase<StreamToHandshakePass> {
 public:
  void runOnOperation() override {
    StreamTypeConverter typeConverter;
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    // Patterns to lower stream dialect operations
    populateStreamToHandshakePatterns(typeConverter, patterns);
    target.addLegalOp<ModuleOp>();
    target.addLegalDialect<handshake::HandshakeDialect>();
    target.addIllegalDialect<func::FuncDialect>();
    target.addIllegalDialect<StreamDialect>();
    // NOTE: we add this here to ensure that the hacky lowerRegion changes
    // will be accepted.
    target.addLegalOp<StreamYieldOp>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
}  // namespace

std::unique_ptr<Pass> mlir::stream::createStreamToHandshakePass() {
  return std::make_unique<StreamToHandshakePass>();
}

