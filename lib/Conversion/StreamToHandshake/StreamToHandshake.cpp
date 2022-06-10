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

#include "circt-stream/Conversion/StreamToHandshake.h"

#include <llvm/ADT/STLExtras.h>

#include "../PassDetail.h"
#include "circt-stream/Dialect/Stream/StreamDialect.h"
#include "circt-stream/Dialect/Stream/StreamOps.h"
#include "circt/Conversion/StandardToHandshake.h"
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Support/SymCache.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace circt::handshake;
using namespace mlir;
using namespace circt_stream;
using namespace circt_stream::stream;

namespace {

/// Returns a name resulting from an operation, without discriminating
/// type information.
static std::string getBareOpName(Operation *op) {
  std::string name = op->getName().getStringRef().str();
  std::replace(name.begin(), name.end(), '.', '_');
  return name;
}

/// Helper class that provides functionality for creating unique symbol names.
/// One instance is shared among all patterns.
class SymbolUniquer : public SymbolCache {
 public:
  SymbolUniquer(Operation *top) : context(top->getContext()) {
    addDefinitions(top);
  }

  mlir::Operation *getDefinition(StringRef str) const {
    auto attr = StringAttr::get(context, str);
    return SymbolCache::getDefinition(attr);
  }

  // TODO: does this have to be thread save?
  std::string getUniqueSymName(Operation *op) {
    std::string opName = getBareOpName(op);
    std::string name = opName;

    unsigned cnt = 1;
    while (getDefinition(name)) {
      name = llvm::formatv("{0}_{1}", opName, cnt++);
    }
    // Note: We do not add the actual op matching the symbol
    auto attr = StringAttr::get(context, name);
    // TODO: dropping the `SymbolCache::` causes a segfault -> investigate
    SymbolCache::addDefinition(attr, op);

    return name;
  }

 private:
  MLIRContext *context;
};

class StreamTypeConverter : public TypeConverter {
 public:
  StreamTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](StreamType type, SmallVectorImpl<Type> &res) {
      MLIRContext *ctx = type.getContext();
      res.push_back(TupleType::get(
          ctx, {type.getElementType(), IntegerType::get(ctx, 1)}));
      res.push_back(NoneType::get(ctx));
      return success();
    });
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
    SmallVector<Type> newResults;
    if (failed(
            typeConverter->convertSignatureArgs(oldFuncType.getInputs(), sig)))
      return failure();

    if (failed(typeConverter->convertTypes(oldFuncType.getResults(),
                                           newResults)) ||
        failed(
            rewriter.convertRegionTypes(&op.getBody(), *typeConverter, &sig)))
      return failure();

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
    newFuncOp.resolveArgAndResNames();

    return success();
  }
};

static Value getBlockCtrlSignal(Block *block) {
  Value ctrl = block->getArguments().back();
  assert(ctrl.getType().isa<NoneType>() &&
         "last argument should be a ctrl signal");
  return ctrl;
}

// TODO: this function require strong assumptions. Relax this if possible
// Assumes that the op producing the input data also produces a ctrl signal
static Value getCtrlSignal(ValueRange operands) {
  assert(operands.size() > 0);
  Value fstOp = operands.front();
  if (BlockArgument arg = fstOp.dyn_cast_or_null<BlockArgument>()) {
    return getBlockCtrlSignal(arg.getOwner());
  }
  Operation *op = fstOp.getDefiningOp();
  if (isa<handshake::InstanceOp>(op)) {
    return op->getResults().back();
  }

  return getCtrlSignal(op->getOperands());
}

static void resolveStreamOperand(Value oldOperand,
                                 SmallVectorImpl<Value> &newOperands) {
  assert(oldOperand.getType().isa<StreamType>());
  // TODO: is there another way to resolve this directly?
  auto castOp =
      dyn_cast<UnrealizedConversionCastOp>(oldOperand.getDefiningOp());
  for (auto castOperand : castOp.getInputs())
    newOperands.push_back(castOperand);
}

static void resolveNewOperands(ValueRange oldOperands,
                               ValueRange remappedOperands,
                               SmallVectorImpl<Value> &newOperands) {
  for (auto [oldOp, remappedOp] : llvm::zip(oldOperands, remappedOperands)) {
    resolveStreamOperand(remappedOp, newOperands);
  }
}

struct ReturnOpLowering : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::ReturnOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands;
    resolveNewOperands(op.getOperands(), adaptor.getOperands(), operands);

    rewriter.replaceOpWithNewOp<handshake::ReturnOp>(op, operands);
    return success();
  }
};

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
static InstanceOp replaceWithInstance(Operation *op, FuncOp func,
                                      ValueRange newOperands,
                                      ConversionPatternRewriter &rewriter) {
  rewriter.setInsertionPoint(op);
  InstanceOp instance =
      rewriter.create<InstanceOp>(op->getLoc(), func, newOperands);

  rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
      op, op->getResultTypes(), instance->getResults());

  return instance;
}

static handshake::UnpackOp unpackAndReplace(
    BlockArgument arg, Location loc, ConversionPatternRewriter &rewriter) {
  assert(arg.getType().isa<TupleType>() && "can only unpack tuple types");
  Block *block = arg.getOwner();
  rewriter.setInsertionPointToStart(block);
  auto unpack = rewriter.create<handshake::UnpackOp>(loc, arg);
  rewriter.replaceUsesOfBlockArgument(arg, unpack.getResult(0));
  return unpack;
}

// Usual flow:
// 1. Apply lowerRegion from StdToHandshake
// 2. Collect operands
// 3. Create new signature
// 4. Apply signature changes
// 5. Change parts of the lowered Region to fit the operations needs.
// 6. Create function and replace operation with InstanceOp

template <typename Op>
struct StreamOpLowering : public OpConversionPattern<Op> {
  StreamOpLowering(TypeConverter &typeConverter, MLIRContext *ctx,
                   SymbolUniquer &symbolUniquer)
      : OpConversionPattern<Op>(typeConverter, ctx),
        symbolUniquer(symbolUniquer) {}

  SymbolUniquer &symbolUniquer;
};

/// This is used instead of the default TypeConverter::convertSignatureArgs
/// because the unrealized conversion cast somehow cause problems.
/// TODO(culmann): try to fix this.
///   Maybe we could use a separate typeConverter combined with argument
///   materializations, but the behaviour is a bit funky.
static LogicalResult convertSignatureArgs(
    TypeConverter *typeConverter, TypeConverter::SignatureConversion &sig,
    TypeRange originalTypes) {
  for (auto it : llvm::enumerate(originalTypes)) {
    unsigned i = it.index();
    Type origType = it.value();
    SmallVector<Type> newTypes;
    if (failed(typeConverter->convertTypes(origType, newTypes)))
      return failure();

    assert(newTypes.size() == 2 &&
           "the converted type should originally be a StreamType");
    sig.addInputs(i, newTypes[0]);
    sig.addInputs(newTypes[1]);
  }
  return success();
}

// Builds a handshake::FuncOp and that represents the mapping funtion. This
// function is then instantiated and connected to its inputs and outputs.
struct MapOpLowering : public StreamOpLowering<MapOp> {
  using StreamOpLowering::StreamOpLowering;

  LogicalResult matchAndRewrite(
      MapOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Region &r = op.getRegion();

    TypeConverter *typeConverter = getTypeConverter();
    TypeConverter::SignatureConversion sig(adaptor.getOperands().size() + 1);

    if (failed(convertSignatureArgs(typeConverter, sig, op->getOperandTypes())))
      return failure();

    //  add the ctrl input
    sig.addInputs(adaptor.getOperands().size(), rewriter.getNoneType());

    Block *entryBlock =
        rewriter.applySignatureConversion(&r, sig, typeConverter);
    // replace old ctrl signal
    rewriter.replaceUsesOfBlockArgument(entryBlock->getArgument(2),
                                        entryBlock->getArgument(1));
    entryBlock->eraseArgument(2);

    auto unpack = unpackAndReplace(entryBlock->getArgument(0), loc, rewriter);
    Value eos = unpack.getResult(1);

    Operation *oldTerm = entryBlock->getTerminator();

    // conversion before that we need a clean binding of EOS values
    rewriter.setInsertionPoint(oldTerm);
    auto tupleOut = rewriter.create<handshake::PackOp>(
        oldTerm->getLoc(), ValueRange({oldTerm->getOperand(0), eos}));

    SmallVector<Value> newTermOperands = {tupleOut, oldTerm->getOperand(1)};
    auto newTerm = rewriter.replaceOpWithNewOp<handshake::ReturnOp>(
        oldTerm, newTermOperands);

    TypeRange resTypes = newTerm->getOperandTypes();

    SmallVector<Value> operands;
    resolveNewOperands(op->getOperands(), adaptor.getOperands(), operands);

    rewriter.setInsertionPointToStart(getTopLevelBlock(op));
    FuncOp newFuncOp =
        createFuncOp(r, symbolUniquer.getUniqueSymName(op),
                     entryBlock->getArgumentTypes(), resTypes, rewriter);

    replaceWithInstance(op, newFuncOp, operands, rewriter);

    return success();
  }
};

struct FilterOpLowering : public StreamOpLowering<FilterOp> {
  using StreamOpLowering::StreamOpLowering;

  LogicalResult matchAndRewrite(
      FilterOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Region &r = op.getRegion();

    TypeConverter *typeConverter = getTypeConverter();
    TypeConverter::SignatureConversion sig(adaptor.getOperands().size() + 1);

    if (failed(typeConverter->convertSignatureArgs(op->getOperandTypes(), sig)))
      return failure();

    //  add the ctrl input
    sig.addInputs(adaptor.getOperands().size(), rewriter.getNoneType());

    // add filtering mechanism
    Block *entryBlock =
        rewriter.applySignatureConversion(&r, sig, typeConverter);

    auto unpack = unpackAndReplace(entryBlock->getArgument(0), loc, rewriter);
    Value input = unpack.getResult(0);
    Value eos = unpack.getResult(1);

    Operation *oldTerm = entryBlock->getTerminator();

    assert(oldTerm->getNumOperands() == 2 &&
           "expected handshake::ReturnOp to have two operands");
    rewriter.setInsertionPointToEnd(entryBlock);

    Value cond = oldTerm->getOperand(0);
    Value ctrl = oldTerm->getOperand(1);

    auto tupleOut =
        rewriter.create<handshake::PackOp>(loc, ValueRange({input, eos}));

    auto condOrEos = rewriter.create<arith::OrIOp>(loc, cond, eos);

    auto dataBr = rewriter.create<handshake::ConditionalBranchOp>(
        rewriter.getUnknownLoc(), condOrEos, tupleOut);

    // Makes sure we only emit Ctrl when data is produced
    auto ctrlBr = rewriter.create<handshake::ConditionalBranchOp>(
        rewriter.getUnknownLoc(), condOrEos, ctrl);

    SmallVector<Value> newTermOperands = {dataBr.trueResult(),
                                          ctrlBr.trueResult()};
    auto newTerm = rewriter.replaceOpWithNewOp<handshake::ReturnOp>(
        oldTerm, newTermOperands);

    SmallVector<Value> operands = llvm::to_vector(adaptor.getOperands());
    operands.push_back(getCtrlSignal(adaptor.getOperands()));

    rewriter.setInsertionPointToStart(getTopLevelBlock(op));
    FuncOp newFuncOp = createFuncOp(r, symbolUniquer.getUniqueSymName(op),
                                    entryBlock->getArgumentTypes(),
                                    newTerm.getOperandTypes(), rewriter);

    replaceWithInstance(op, newFuncOp, operands, rewriter);
    return success();
  }
};

/// Lowers a reduce operation to a ahndshake circuit
///
/// Accumulates the result of the reduction in a buffer. On EOS this result is
/// emitted, followed by a EOS = true one cycle after the emission of the
/// result.
///
/// While the reduction is running, no output is produced.
struct ReduceOpLowering : public StreamOpLowering<ReduceOp> {
  using StreamOpLowering::StreamOpLowering;

  LogicalResult matchAndRewrite(
      ReduceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Region &r = op.getRegion();

    TypeConverter *typeConverter = getTypeConverter();
    Type tupleType = typeConverter->convertType(op.result().getType());
    assert(tupleType.isa<TupleType>());
    Type resultType = tupleType.dyn_cast<TupleType>().getType(0);

    // TODO: handshake currently only supports i64 buffers, change this as
    // soon as support for other types is added.
    assert(resultType == rewriter.getI64Type() &&
           "currently, only i64 buffers are supported");

    TypeConverter::SignatureConversion sig(adaptor.getOperands().size() + 2);
    sig.addInputs(0, resultType);
    if (failed(
            typeConverter->convertSignatureArgs(op->getOperandTypes(), sig, 1)))
      return failure();

    //  add the ctrl input
    sig.addInputs(adaptor.getOperands().size() + 1, rewriter.getNoneType());

    Block *entryBlock =
        rewriter.applySignatureConversion(&r, sig, typeConverter);

    auto unpack = unpackAndReplace(entryBlock->getArgument(1), loc, rewriter);
    Value eosIn = unpack.getResult(1);

    Operation *oldTerm = entryBlock->getTerminator();
    rewriter.setInsertionPoint(oldTerm);
    auto buffer = rewriter.create<handshake::BufferOp>(
        rewriter.getUnknownLoc(), resultType, 1, oldTerm->getOperand(0),
        BufferTypeEnum::seq);

    // This does return an unsigned integer but expects signed integers
    // TODO check if this is an MLIR bug
    buffer->setAttr("initValues",
                    rewriter.getI64ArrayAttr({(int64_t)adaptor.initValue()}));

    assert(eosIn.getType() == rewriter.getI1Type());

    auto dataBr = rewriter.create<handshake::ConditionalBranchOp>(
        rewriter.getUnknownLoc(), eosIn, buffer);
    auto eosBr = rewriter.create<handshake::ConditionalBranchOp>(
        rewriter.getUnknownLoc(), eosIn, eosIn);
    auto ctrlBr = rewriter.create<handshake::ConditionalBranchOp>(
        rewriter.getUnknownLoc(), eosIn, oldTerm->getOperand(1));

    rewriter.replaceUsesOfBlockArgument(entryBlock->getArgument(0),
                                        dataBr.falseResult());
    entryBlock->eraseArgument(0);

    // Connect outputs and ensure correct delay between value and EOS=true
    // emission A sequental buffer ensures a cycle delay of 1
    auto eosFalse = rewriter.create<handshake::ConstantOp>(
        rewriter.getUnknownLoc(),
        rewriter.getIntegerAttr(rewriter.getI1Type(), 0), ctrlBr.trueResult());
    auto tupleOutVal = rewriter.create<handshake::PackOp>(
        loc, ValueRange({dataBr.trueResult(), eosFalse}));

    auto tupleOutEOS = rewriter.create<handshake::PackOp>(
        loc, ValueRange({dataBr.trueResult(), eosBr.trueResult()}));

    // Not really needed, but the BufferOp builder requires an input
    auto bubble = rewriter.create<ConstantOp>(
        loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0),
        ctrlBr.trueResult());
    auto select = rewriter.create<handshake::BufferOp>(
        rewriter.getUnknownLoc(), rewriter.getI32Type(), 2, bubble,
        BufferTypeEnum::seq);
    // First select the tupleOut, afterwards the one with the EOS signal
    select->setAttr("initValues", rewriter.getI64ArrayAttr({1, 0}));

    auto tupleOut = rewriter.create<MuxOp>(
        loc, select, ValueRange({tupleOutVal, tupleOutEOS}));
    auto ctrlOut = rewriter.create<MuxOp>(
        loc, select, ValueRange({ctrlBr.trueResult(), ctrlBr.trueResult()}));

    SmallVector<Value> newTermOperands = {tupleOut, ctrlOut};

    auto newTerm = rewriter.replaceOpWithNewOp<handshake::ReturnOp>(
        oldTerm, newTermOperands);

    SmallVector<Value> operands = llvm::to_vector(adaptor.getOperands());
    operands.push_back(getCtrlSignal(adaptor.getOperands()));

    rewriter.setInsertionPointToStart(getTopLevelBlock(op));
    FuncOp newFuncOp = createFuncOp(r, symbolUniquer.getUniqueSymName(op),
                                    entryBlock->getArgumentTypes(),
                                    newTerm.getOperandTypes(), rewriter);

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

struct CreateOpLowering : public StreamOpLowering<CreateOp> {
  using StreamOpLowering::StreamOpLowering;

  // TODO add location usage
  LogicalResult matchAndRewrite(
      stream::CreateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Region r;
    Location loc = op.getLoc();

    Block *entryBlock = rewriter.createBlock(&r, {}, {rewriter.getNoneType()},
                                             {rewriter.getUnknownLoc()});

    // TODO ensure that subsequent ctrl inputs are ignored
    Value ctrlIn = entryBlock->getArgument(0);
    size_t bufSize = op.values().size();
    Type elementType = op.getElementType();
    assert(elementType.isa<IntegerType>());

    rewriter.setInsertionPointToEnd(entryBlock);

    // Only use in ctrl once
    auto falseVal = rewriter.create<handshake::ConstantOp>(
        rewriter.getUnknownLoc(),
        rewriter.getIntegerAttr(rewriter.getI1Type(), 0), ctrlIn);
    auto fst = rewriter.create<BufferOp>(loc, rewriter.getI1Type(), 1, falseVal,
                                         BufferTypeEnum::seq);
    fst->setAttr("initValues", rewriter.getI64ArrayAttr({1}));
    auto useCtrl =
        rewriter.create<handshake::ConditionalBranchOp>(loc, fst, ctrlIn);

    // Ctrl "looping" and selection
    // We have to change the input later on
    auto tmpCtrl = rewriter.create<NeverOp>(loc, rewriter.getNoneType());
    auto ctrlBuf = rewriter.create<BufferOp>(loc, rewriter.getNoneType(), 1,
                                             tmpCtrl, BufferTypeEnum::seq);
    auto ctrl = rewriter.create<MergeOp>(
        loc, ValueRange({useCtrl.trueResult(), ctrlBuf}));
    rewriter.replaceOp(tmpCtrl, {ctrl});

    // Data part

    auto bubble = rewriter.create<handshake::ConstantOp>(
        loc, rewriter.getIntegerAttr(elementType, 0), ctrl);
    auto dataBuf = rewriter.create<BufferOp>(loc, elementType, bufSize, bubble,
                                             BufferTypeEnum::seq);
    // The buffer works in reverse
    SmallVector<int64_t> values;
    for (auto attr : llvm::reverse(op.values())) {
      assert(attr.isa<IntegerAttr>());
      values.push_back(attr.dyn_cast<IntegerAttr>().getInt());
    }
    dataBuf->setAttr("initValues", rewriter.getI64ArrayAttr(values));
    auto cnt = rewriter.create<BufferOp>(loc, rewriter.getI64Type(), 1, bubble,
                                         BufferTypeEnum::seq);
    // initialize cnt to 0 to indicate that 0 elements were emitted
    cnt->setAttr("initValues", rewriter.getI64ArrayAttr({0}));

    auto one = rewriter.create<handshake::ConstantOp>(
        loc, rewriter.getIntegerAttr(rewriter.getI64Type(), 1), ctrl);

    auto sizeConst = rewriter.create<handshake::ConstantOp>(
        loc, rewriter.getIntegerAttr(rewriter.getI64Type(), bufSize), ctrl);

    auto finished = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, cnt, sizeConst);

    auto newCnt = rewriter.create<arith::AddIOp>(op.getLoc(), cnt, one);
    // ensure looping of cnt
    cnt.setOperand(newCnt);

    auto tupleOut = rewriter.create<handshake::PackOp>(
        loc, ValueRange({dataBuf, finished}));

    // create terminator
    auto term = rewriter.create<handshake::ReturnOp>(
        loc, ValueRange({tupleOut.result(), ctrl}));

    // Collect types of function
    SmallVector<Type> argTypes;
    argTypes.push_back(rewriter.getNoneType());

    rewriter.setInsertionPointToStart(getTopLevelBlock(op));
    auto newFuncOp = createFuncOp(r, symbolUniquer.getUniqueSymName(op),
                                  argTypes, term.getOperandTypes(), rewriter);

    replaceWithInstance(op, newFuncOp, {getBlockCtrlSignal(op->getBlock())},
                        rewriter);
    return success();
  }
};

struct SplitOpLowering : public StreamOpLowering<SplitOp> {
  using StreamOpLowering::StreamOpLowering;

  LogicalResult matchAndRewrite(
      SplitOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Region &r = op.getRegion();

    TypeConverter *typeConverter = getTypeConverter();
    TypeConverter::SignatureConversion sig(adaptor.getOperands().size() + 1);

    if (failed(typeConverter->convertSignatureArgs(op->getOperandTypes(), sig)))
      return failure();

    //  add the ctrl input
    sig.addInputs(adaptor.getOperands().size(), rewriter.getNoneType());

    Block *entryBlock =
        rewriter.applySignatureConversion(&r, sig, typeConverter);

    auto unpack = unpackAndReplace(entryBlock->getArgument(0), loc, rewriter);
    Value eos = unpack.getResult(1);

    Operation *oldTerm = entryBlock->getTerminator();

    rewriter.setInsertionPoint(oldTerm);
    SmallVector<Value> newTermOperands = llvm::to_vector(
        llvm::map_range(oldTerm->getOperands().drop_back(), [&](auto oldOp) {
          return rewriter
              .create<handshake::PackOp>(oldTerm->getLoc(),
                                         ValueRange({oldOp, eos}))
              .getResult();
        }));

    newTermOperands.push_back(oldTerm->getOperands().back());
    auto newTerm = rewriter.replaceOpWithNewOp<handshake::ReturnOp>(
        oldTerm, newTermOperands);

    TypeRange resTypes = newTerm->getOperandTypes();

    SmallVector<Value> operands = llvm::to_vector(adaptor.getOperands());
    operands.push_back(getCtrlSignal(adaptor.getOperands()));

    rewriter.setInsertionPointToStart(getTopLevelBlock(op));
    FuncOp newFuncOp =
        createFuncOp(r, symbolUniquer.getUniqueSymName(op),
                     entryBlock->getArgumentTypes(), resTypes, rewriter);

    replaceWithInstance(op, newFuncOp, operands, rewriter);

    return success();
  }
};

static void populateStreamToHandshakePatterns(
    StreamTypeConverter &typeConverter, SymbolUniquer symbolUniquer,
    RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
    FuncOpLowering,
    ReturnOpLowering,
    PackOpLowering,
    UnpackOpLowering
  >(typeConverter, patterns.getContext());

  patterns.add<
    MapOpLowering,
    FilterOpLowering,
    ReduceOpLowering,
    CreateOpLowering,
    SplitOpLowering
  >(typeConverter, patterns.getContext(), symbolUniquer);
  // clang-format on
}

// ensures that the IR is in a valid state after the initial partial
// conversion
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

// TODO Do this with an op trait?
bool isStreamOp(Operation *op) {
  return isa<MapOp, FilterOp, ReduceOp, SplitOp>(op);
}

/// Traverses the modules region recursively and applies the std to handshake
/// conversion on each stream operation region.
LogicalResult transformStdRegions(ModuleOp m) {
  // go over all stream ops and transform their regions
  for (auto funcOp : llvm::make_early_inc_range(m.getOps<func::FuncOp>())) {
    if (funcOp.isDeclaration()) continue;
    Region *funcRegion = funcOp.getCallableRegion();
    for (Operation &op : funcRegion->getOps()) {
      if (!isStreamOp(&op)) continue;
      for (auto &r : op.getRegions()) {
        StreamLowering sl(r);
        if (failed(lowerRegion<YieldOp>(sl, false, false))) return failure();
      }
    }
  }
  return success();
}

static LogicalResult removeUnusedConversionCasts(ModuleOp m) {
  for (auto funcOp : m.getOps<handshake::FuncOp>()) {
    if (funcOp.isDeclaration()) continue;
    Region &funcRegion = funcOp.body();
    for (auto op : funcRegion.getOps<UnrealizedConversionCastOp>()) {
      op->erase();
    }
  }
  return success();
}

class StreamToHandshakePass
    : public StreamToHandshakeBase<StreamToHandshakePass> {
 public:
  void runOnOperation() override {
    if (failed(transformStdRegions(getOperation()))) {
      signalPassFailure();
      return;
    }
    StreamTypeConverter typeConverter;
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    SymbolUniquer symbolUniquer(getOperation());

    // Patterns to lower stream dialect operations
    populateStreamToHandshakePatterns(typeConverter, symbolUniquer, patterns);
    target.addLegalOp<ModuleOp>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addLegalDialect<handshake::HandshakeDialect>();
    target.addLegalDialect<arith::ArithmeticDialect>();
    target.addIllegalDialect<func::FuncDialect>();
    target.addIllegalDialect<StreamDialect>();
    // NOTE: we add this here to ensure that the hacky lowerRegion changes
    // will be accepted.
    target.addLegalOp<YieldOp>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();

    if (failed(removeUnusedConversionCasts(getOperation())))
      signalPassFailure();

    if (failed(materializeForksAndSinks(getOperation()))) signalPassFailure();
  }
};
}  // namespace

std::unique_ptr<Pass> circt_stream::createStreamToHandshakePass() {
  return std::make_unique<StreamToHandshakePass>();
}

