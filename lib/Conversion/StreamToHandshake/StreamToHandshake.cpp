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

#include <llvm/ADT/STLExtras.h>

#include "../PassDetail.h"
#include "Standalone/Dialect/Stream/StreamDialect.h"
#include "Standalone/Dialect/Stream/StreamOps.h"
#include "circt/Conversion/StandardToHandshake.h"
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
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
    if (failed(
            typeConverter->convertSignatureArgs(oldFuncType.getInputs(), sig)))
      return failure();

    // TODO replace the above with custom functionality to fill up the signature
    // conversion Currently, EOS is only added at the end of the inputs and not
    // directly after each streaming value

    // Add EOS for each stream input
    for (auto it : llvm::enumerate(oldFuncType.getInputs())) {
      auto type = it.value();
      if (!type.isa<StreamType>()) continue;
      sig.addInputs(IntegerType::get(type.getContext(), 1));
    }

    // add the ctrl input
    sig.addInputs({rewriter.getNoneType()});
    if (failed(typeConverter->convertTypes(oldFuncType.getResults(),
                                           newResults)) ||
        failed(
            rewriter.convertRegionTypes(&op.getBody(), *typeConverter, &sig)))
      return failure();

    // TODO same problem as for the input types

    // Add result EOS types
    for (auto it : llvm::enumerate(oldFuncType.getInputs())) {
      auto type = it.value();
      if (!type.isa<StreamType>()) continue;
      newResults.push_back(IntegerType::get(type.getContext(), 1));
    }

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

// TODO: this function require strong assumptions. Relax this if possible
// Assumes that the op producing the input data also produces a ctrl signal
static Value getCtrlSignal(ValueRange operands) {
  assert(operands.size() > 0);
  Value fstOp = operands.front();
  if (BlockArgument arg = fstOp.dyn_cast_or_null<BlockArgument>()) {
    Block *block = arg.getOwner();
    return block->getArguments().back();
  }
  Operation *op = fstOp.getDefiningOp();
  if (isa<handshake::InstanceOp>(op)) {
    return op->getResults().back();
  }

  return getCtrlSignal(op->getOperands());
}

/// Searches the EOS signal corresponding to the operand
static Value getEOSSignal(Value operand) {
  if (auto arg = operand.dyn_cast<BlockArgument>()) {
    Block *block = arg.getParentBlock();
    unsigned argNum = arg.getArgNumber() + 1;
    assert(argNum < block->getNumArguments());
    return block->getArgument(argNum);
  }

  auto res = operand.dyn_cast<OpResult>();
  Operation *defOp = res.getDefiningOp();

  unsigned resNum = res.getResultNumber() + 1;
  assert(resNum < defOp->getNumResults());
  return defOp->getResult(resNum);
}

struct ReturnOpLowering : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::ReturnOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands = llvm::to_vector(adaptor.getOperands());

    assert(operands.size() == 1 &&
           "currently multiple streams are not supported");
    // Add EOS signal
    operands.push_back(getEOSSignal(operands[0]));

    // return block arg ctrl signal if nothing else has to be returned
    if (adaptor.getOperands().size() == 0) {
      Value ctrl = op->getBlock()->getArguments().back();
      operands.push_back(ctrl);
    } else {
      operands.push_back(getCtrlSignal(adaptor.getOperands()));
    }
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
  // Look into the factility CIRCT provides
  return opName;
}

static Block *getTopLevelBlock(Operation *op) {
  return &op->getParentOfType<ModuleOp>().getRegion().front();
}

/// Creates a new FuncOp that encapsulates the provided region.
static FuncOp createFuncOp(Region &region, StringRef name, TypeRange argTypes,
                           TypeRange resTypes,
                           ConversionPatternRewriter &rewriter) {
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
  rewriter.setInsertionPoint(op);
  InstanceOp instance =
      rewriter.create<InstanceOp>(op->getLoc(), func, newOperands);

  // The ctrl and EOS signal has to be ignored
  rewriter.replaceOp(op, instance.getResults().front());
}

static void collectOperandsAndSignature(SmallVectorImpl<Value> &newOperands,
                                        ValueRange adaptorOperands,
                                        TypeRange oldTypes,
                                        TypeConverter::SignatureConversion &sig,
                                        MLIRContext *ctx,
                                        unsigned sigOffset = 0) {
  assert(adaptorOperands.size() == oldTypes.size());
  IntegerType i1Type = IntegerType::get(ctx, 1);

  for (auto it : llvm::enumerate(llvm::zip(adaptorOperands, oldTypes))) {
    size_t i = it.index() + sigOffset;
    auto [operand, oldType] = it.value();

    newOperands.push_back(operand);

    // Collect new input types
    sig.addInputs(i, operand.getType());

    // Add EOS signals for al stream inputs
    if (oldType.isa<StreamType>()) {
      sig.addInputs(i1Type);

      newOperands.push_back(getEOSSignal(operand));
    }
  }

  // add missing NoneType
  sig.addInputs(adaptorOperands.size() + sigOffset, NoneType::get(ctx));
  newOperands.push_back(getCtrlSignal(adaptorOperands));
}

// Usual flow:
// 1. Apply lowerRegion from StdToHandshake
// 2. Collect operands
// 3. Create new signature
// 4. Apply signature changes
// 5. Change parts of the lowered Region to fit the operations needs.
// 6. Create function and replace operation with InstanceOp

// Builds a handshake::FuncOp and that represents the mapping funtion. This
// function is then instantiated and connected to its inputs and outputs.
struct MapOpLowering : public OpConversionPattern<MapOp> {
  using OpConversionPattern<MapOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MapOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Region &r = op.getRegion();
    StreamLowering sl(r);
    if (failed(lowerRegion<YieldOp>(sl, false, false))) return failure();

    TypeConverter *typeConverter = getTypeConverter();

    SmallVector<Value> operands;
    TypeConverter::SignatureConversion sig(adaptor.getOperands().size() + 1);

    collectOperandsAndSignature(operands, adaptor.getOperands(),
                                op->getOperandTypes(), sig,
                                rewriter.getContext());

    // TODO EOS can overtake in-flight tuples
    Block *entryBlock =
        rewriter.applySignatureConversion(&r, sig, typeConverter);
    Operation *oldTerm = entryBlock->getTerminator();

    // conversion before that we need a clean binding of EOS values
    SmallVector<Value> newTermOperands = {oldTerm->getOperand(0),
                                          entryBlock->getArgument(1),
                                          oldTerm->getOperand(1)};
    rewriter.setInsertionPoint(oldTerm);
    auto newTerm = rewriter.replaceOpWithNewOp<handshake::ReturnOp>(
        oldTerm, newTermOperands);

    TypeRange resTypes = newTerm->getOperandTypes();

    rewriter.setInsertionPointToStart(getTopLevelBlock(op));
    FuncOp newFuncOp = createFuncOp(
        r, getFuncName(op), entryBlock->getArgumentTypes(), resTypes, rewriter);

    replaceWithInstance(op, newFuncOp, operands, rewriter);

    return success();
  }
};

struct FilterOpLowering : public OpConversionPattern<FilterOp> {
  using OpConversionPattern<FilterOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      FilterOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Region &r = op.getRegion();
    StreamLowering sl(r);
    if (failed(lowerRegion<YieldOp>(sl, false, false))) return failure();

    TypeConverter *typeConverter = getTypeConverter();

    SmallVector<Value> operands;
    TypeConverter::SignatureConversion sig(adaptor.getOperands().size() + 1);

    collectOperandsAndSignature(operands, adaptor.getOperands(),
                                op->getOperandTypes(), sig,
                                rewriter.getContext());

    // add filtering mechanism
    Block *entryBlock =
        rewriter.applySignatureConversion(&r, sig, typeConverter);
    Operation *oldTerm = entryBlock->getTerminator();

    assert(oldTerm->getNumOperands() == 2 &&
           "expected handshake::ReturnOp to have two operands");
    rewriter.setInsertionPointToEnd(entryBlock);
    Value input = entryBlock->getArgument(0);
    Value cond = oldTerm->getOperand(0);
    Value ctrl = oldTerm->getOperand(1);

    auto condBr = rewriter.create<handshake::ConditionalBranchOp>(
        rewriter.getUnknownLoc(), cond, input);

    // TODO EOS can overtake in-flight tuples
    SmallVector<Value> newTermOperands = {condBr.trueResult(),
                                          entryBlock->getArgument(1), ctrl};
    auto newTerm = rewriter.replaceOpWithNewOp<handshake::ReturnOp>(
        oldTerm, newTermOperands);

    TypeRange resTypes = newTerm.getOperandTypes();

    rewriter.setInsertionPointToStart(getTopLevelBlock(op));
    FuncOp newFuncOp = createFuncOp(
        r, getFuncName(op), entryBlock->getArgumentTypes(), resTypes, rewriter);

    replaceWithInstance(op, newFuncOp, operands, rewriter);
    return success();
  }
};

struct ReduceOpLowering : public OpConversionPattern<ReduceOp> {
  using OpConversionPattern<ReduceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ReduceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Region &r = op.getRegion();
    StreamLowering sl(r);
    if (failed(lowerRegion<YieldOp>(sl, false, false))) return failure();

    TypeConverter *typeConverter = getTypeConverter();
    Type resultType = typeConverter->convertType(op.res().getType());

    // TODO: handshake currently only supports i64 buffers, change this as soon
    // as support for other types is added.
    assert(resultType == rewriter.getI64Type() &&
           "currently, only i64 buffers are supported");

    SmallVector<Value> operands;
    TypeConverter::SignatureConversion sig(adaptor.getOperands().size() + 2);
    sig.addInputs(0, resultType);

    // Signature is incompatible with the amount of operands, thus offset of 1
    collectOperandsAndSignature(operands, adaptor.getOperands(),
                                op->getOperandTypes(), sig,
                                rewriter.getContext(), 1);

    Block *entryBlock =
        rewriter.applySignatureConversion(&r, sig, typeConverter);
    // Block *entryBlock = &r.front();
    Operation *oldTerm = entryBlock->getTerminator();
    rewriter.setInsertionPointToStart(entryBlock);
    auto buffer = rewriter.create<handshake::BufferOp>(
        rewriter.getUnknownLoc(), resultType, 1, oldTerm->getOperand(0),
        BufferTypeEnum::seq);

    // This does return an unsigned integer but expects signed integers
    // TODO check if this is an MLIR bug
    buffer->setAttr("initValues",
                    rewriter.getI64ArrayAttr({(int64_t)adaptor.initValue()}));

    Value eos = entryBlock->getArgument(2);
    assert(eos.getType() == rewriter.getI1Type());
    auto condBr = rewriter.create<handshake::ConditionalBranchOp>(
        rewriter.getUnknownLoc(), eos, buffer);

    // Buffer -> either feed to lambda or to output, depending on EOS
    // signal. Forward EOS to output
    rewriter.replaceUsesOfBlockArgument(entryBlock->getArgument(0),
                                        condBr.falseResult());
    entryBlock->eraseArgument(0);

    // TODO we need a clean binding of EOS values
    SmallVector<Value> newTermOperands = {condBr.trueResult(),
                                          entryBlock->getArgument(1),
                                          oldTerm->getOperand(1)};
    rewriter.setInsertionPoint(oldTerm);
    auto newTerm = rewriter.replaceOpWithNewOp<handshake::ReturnOp>(
        oldTerm, newTermOperands);

    TypeRange resTypes = newTerm.getOperandTypes();
    rewriter.setInsertionPointToStart(getTopLevelBlock(op));
    FuncOp newFuncOp = createFuncOp(
        r, getFuncName(op), entryBlock->getArgumentTypes(), resTypes, rewriter);

    replaceWithInstance(op, newFuncOp, operands, rewriter);
    return success();
  }
};

struct PackOpLowering : public OpConversionPattern<stream::PackOp> {
  using OpConversionPattern<stream::PackOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      stream::PackOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<handshake::PackOp>(op, adaptor.getOperands());

    return success();
  }
};

struct UnpackOpLowering : public OpConversionPattern<stream::UnpackOp> {
  using OpConversionPattern<stream::UnpackOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      stream::UnpackOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<handshake::UnpackOp>(op, adaptor.input());

    return success();
  }
};

static void populateStreamToHandshakePatterns(
    StreamTypeConverter &typeConverter, RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
    FuncOpLowering,
    ReturnOpLowering,
    MapOpLowering,
    FilterOpLowering,
    ReduceOpLowering,
    PackOpLowering,
    UnpackOpLowering
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
    target.addLegalOp<YieldOp>();

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

