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

// Assumes that the op producing the input data also produces a ctrl signal
static Value getCtrlSignal(ValueRange operands) {
  assert(operands.size() > 0);
  Value fstOp = operands.front();
  Value ctrl;
  if (BlockArgument arg = fstOp.dyn_cast_or_null<BlockArgument>()) {
    Block *block = arg.getOwner();
    ctrl = block->getArguments().back();
  } else {
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

    Value ctrl = getCtrlSignal(adaptor.getOperands());
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
  // TODO add unique
  return opName;
}

static Block *getTopLevelBock(Operation *op) {
  return &op->getParentOfType<ModuleOp>().getRegion().front();
}

/// Creates a new FuncOp that encapsulates the provided region.
static FuncOp createFuncOp(Region &region, StringRef name,
                           SmallVectorImpl<mlir::Type> &argTypes,
                           SmallVectorImpl<mlir::Type> &resTypes,
                           ConversionPatternRewriter &rewriter) {
  auto noneType = rewriter.getNoneType();
  argTypes.push_back(noneType);
  resTypes.push_back(noneType);

  auto func_type = rewriter.getFunctionType(argTypes, resTypes);
  FuncOp newFuncOp =
      rewriter.create<FuncOp>(rewriter.getUnknownLoc(), name, func_type);

  // Makes the function private
  SymbolTable::setSymbolVisibility(newFuncOp, SymbolTable::Visibility::Private);

  rewriter.inlineRegionBefore(region, newFuncOp.getBody(), newFuncOp.end());
  newFuncOp.resolveArgAndResNames();
  assert(newFuncOp.getRegion().hasOneBlock() &&
         "expected std to handshake to produce region with one block");

  return newFuncOp;
}

/// Replaces op with a new InstanceOp that calls the provided function.
static void replaceWithInstance(Operation *op, FuncOp func,
                                ValueRange newOperands,
                                ConversionPatternRewriter &rewriter) {
  SmallVector<Value, 2> operands;
  for (auto operand : newOperands) operands.push_back(operand);

  operands.push_back(getCtrlSignal(newOperands));

  rewriter.setInsertionPoint(op);
  InstanceOp instance =
      rewriter.create<InstanceOp>(op->getLoc(), func, operands);

  // The ctrl signal has to be ignored
  rewriter.replaceOp(op, instance.getResults().drop_back());
}

// Builds a handshake::FuncOp and that represents the mapping funtion. This
// function is then instantiated and connected to its inputs and outputs.
struct StreamMapLowering : public OpConversionPattern<StreamMap> {
  using OpConversionPattern<StreamMap>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      StreamMap op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    TypeConverter *typeConverter = getTypeConverter();

    StreamLowering sl(op.getRegion());

    if (failed(lowerRegion<StreamYieldOp>(sl, false, false))) return failure();

    SmallVector<mlir::Type, 8> argTypes = {
        typeConverter->convertType(op.input().getType())};

    SmallVector<mlir::Type, 8> resTypes = {
        typeConverter->convertType(op.res().getType())};

    rewriter.setInsertionPointToStart(getTopLevelBock(op));
    FuncOp newFuncOp = createFuncOp(op.getRegion(), getFuncName(op), argTypes,
                                    resTypes, rewriter);

    replaceWithInstance(op, newFuncOp, adaptor.getOperands(), rewriter);

    return success();
  }
};

struct StreamFilterLowering : public OpConversionPattern<StreamFilter> {
  using OpConversionPattern<StreamFilter>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      StreamFilter op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    TypeConverter *typeConverter = getTypeConverter();

    StreamLowering sl(op.getRegion());

    if (failed(lowerRegion<StreamYieldOp>(sl, false, false))) return failure();

    SmallVector<mlir::Type, 8> argTypes = {
        typeConverter->convertType(op.input().getType())};

    SmallVector<mlir::Type, 8> resTypes = {
        typeConverter->convertType(op.input().getType())};

    rewriter.setInsertionPointToStart(getTopLevelBock(op));
    FuncOp newFuncOp = createFuncOp(op.getRegion(), getFuncName(op), argTypes,
                                    resTypes, rewriter);

    // add filtering mechanism
    Block *entryBlock = &newFuncOp.getRegion().front();
    Operation *oldTerm = entryBlock->getTerminator();

    assert(oldTerm->getNumOperands() == 2 &&
           "expected handshake::ReturnOp to have two operands");
    rewriter.setInsertionPointToEnd(entryBlock);
    Value input = entryBlock->getArgument(0);
    Value cond = oldTerm->getOperand(0);
    Value ctrl = oldTerm->getOperand(1);

    auto condBr = rewriter.create<handshake::ConditionalBranchOp>(
        rewriter.getUnknownLoc(), cond, input);

    SmallVector<Value, 2> newOperands = {condBr.trueResult(), ctrl};
    rewriter.replaceOpWithNewOp<handshake::ReturnOp>(oldTerm, newOperands);

    replaceWithInstance(op, newFuncOp, adaptor.getOperands(), rewriter);

    return success();
  }
};

struct StreamReduceLowering : public OpConversionPattern<StreamReduce> {
  using OpConversionPattern<StreamReduce>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      StreamReduce op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    TypeConverter *typeConverter = getTypeConverter();

    Type resultType = typeConverter->convertType(op.res().getType());

    StreamLowering sl(op.getRegion());

    if (failed(lowerRegion<StreamYieldOp>(sl, false, false))) return failure();

    // TODO: handshake currently only supports i64 buffers, change this as soon
    // as support for other types is added.
    SmallVector<mlir::Type, 8> argTypes = {
        typeConverter->convertType(op.input().getType()),
        rewriter.getI64Type()};

    SmallVector<mlir::Type, 8> resTypes = {
        typeConverter->convertType(op.input().getType())};

    rewriter.setInsertionPointToStart(getTopLevelBock(op));
    FuncOp newFuncOp = createFuncOp(op.getRegion(), getFuncName(op), argTypes,
                                    resTypes, rewriter);

    Block *entryBlock = &newFuncOp.getRegion().front();
    Operation *term = entryBlock->getTerminator();
    rewriter.setInsertionPointToStart(entryBlock);
    auto buffer = rewriter.create<handshake::BufferOp>(
        rewriter.getUnknownLoc(), resultType, 1, term->getOperand(0),
        BufferTypeEnum::seq);
    // This does return an unsigned integer but expects signed integers
    // TODO check if this is an MLIR bug
    buffer->setAttr("initValues",
                    rewriter.getI64ArrayAttr({(int64_t)adaptor.initValue()}));
    rewriter.replaceUsesOfBlockArgument(entryBlock->getArgument(1),
                                        buffer.result());

    replaceWithInstance(op, newFuncOp, adaptor.getOperands(), rewriter);
    return success();
  }
};

static void populateStreamToHandshakePatterns(
    StreamTypeConverter &typeConverter, RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
    FuncOpLowering,
    ReturnOpLowering,
    StreamMapLowering,
    StreamFilterLowering,
    StreamReduceLowering
  >(typeConverter, patterns.getContext());
  // clang-format on
}

// ensures that the IR is in a valid state after the initial partial conversion
static LogicalResult materializeForksAndSinks(ModuleOp m) {
  for (auto funcOp :
       llvm::make_early_inc_range(m.getOps<handshake::FuncOp>())) {
    OpBuilder builder(funcOp);
    if (addForkOps(funcOp.getRegion(), builder).failed() ||
        addSinkOps(funcOp.getRegion(), builder).failed() ||
        verifyAllValuesHasOneUse(funcOp).failed())
      return failure();
  }

  return success();
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

    if (failed(materializeForksAndSinks(getOperation()))) signalPassFailure();
  }
};
}  // namespace

std::unique_ptr<Pass> mlir::stream::createStreamToHandshakePass() {
  return std::make_unique<StreamToHandshakePass>();
}

