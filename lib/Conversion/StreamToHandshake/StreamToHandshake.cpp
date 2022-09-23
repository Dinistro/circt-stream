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
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/SymCache.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

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
/// The uniquer remembers all symbols and new ones by checking that they do not
/// exist yet.
class SymbolUniquer {
public:
  SymbolUniquer(Operation *top) : context(top->getContext()) {
    addDefinitions(top);
  }

  void addDefinitions(mlir::Operation *top) {
    for (auto &region : top->getRegions())
      for (auto &block : region.getBlocks())
        for (auto symOp : block.getOps<mlir::SymbolOpInterface>())
          addSymbol(symOp.getName().str());
  }

  std::string getUniqueSymName(Operation *op) {
    std::string opName = getBareOpName(op);
    std::string name = opName;

    unsigned cnt = 1;
    while (usedNames.contains(name)) {
      name = llvm::formatv("{0}_{1}", opName, cnt++);
    }
    addSymbol(name);

    return name;
  }

  void addSymbol(std::string name) { usedNames.insert(name); }

private:
  MLIRContext *context;
  llvm::SmallSet<std::string, 8> usedNames;
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

  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor adapter,
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
  // TODO: only if we find a way to replace one value by multiple -> check in
  // discourse
  auto castOp =
      dyn_cast<UnrealizedConversionCastOp>(oldOperand.getDefiningOp());
  for (auto castOperand : castOp.getInputs())
    newOperands.push_back(castOperand);
}

static void resolveNewOperands(Operation *oldOperation,
                               ValueRange remappedOperands,
                               SmallVectorImpl<Value> &newOperands) {
  for (auto [oldOp, remappedOp] :
       llvm::zip(oldOperation->getOperands(), remappedOperands))
    resolveStreamOperand(remappedOp, newOperands);
}

struct ReturnOpLowering : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands;
    resolveNewOperands(op, adaptor.getOperands(), operands);

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

  SmallVector<Value> newValues;
  auto resultIt = instance->getResults().begin();
  for (auto oldResType : op->getResultTypes()) {
    assert(oldResType.isa<StreamType>() &&
           "can currently only replace stream types");

    // TODO this is very fragile
    auto tuple = *resultIt++;
    auto ctrl = *resultIt++;

    auto castOp = rewriter.create<UnrealizedConversionCastOp>(
        op->getLoc(), oldResType, ValueRange({tuple, ctrl}));
    newValues.push_back(castOp.getResult(0));
  }
  rewriter.replaceOp(op, newValues);

  return instance;
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

/// Helper function that collects the required info to build registers.
/// TODO: once buffers support other initial values, this has to be added here.
template <typename TOp>
static LogicalResult getRegInfos(TOp op,
                                 SmallVectorImpl<IntegerAttr> &regTypes) {
  Optional<ArrayAttr> regTypeAttr = op.getRegisters();
  if (regTypeAttr.has_value()) {
    for (auto attr : *regTypeAttr) {
      auto res = llvm::TypeSwitch<Attribute, LogicalResult>(attr)
                     .Case<IntegerAttr>([&](auto intAttr) {
                       regTypes.push_back(intAttr);
                       return success();
                     })
                     .Default([&](Attribute) {
                       return op->emitError("unsupported register type");
                     });
      if (failed(res))
        // TODO: this should be checked by a verifier
        return failure();
    }
  }
  return success();
}

/// A helper class to build registers. Tracks the state for building backedges
/// correctly.
class RegisterBuilder {
  RegisterBuilder(ConversionPatternRewriter &rewriter)
      : rewriter(rewriter), backBuilder(rewriter, rewriter.getUnknownLoc()) {}

public:
  template <typename TOp>
  static FailureOr<RegisterBuilder> create(ConversionPatternRewriter &rewriter,
                                           TOp op) {
    RegisterBuilder regBuilder(rewriter);
    if (failed(getRegInfos(op, regBuilder.regInfos)))
      return failure();

    return regBuilder;
  }

private:
  // TODO initialValue type: handshake buffers do currently only support
  // integers
  BufferOp buildRegister(Type type, int64_t initialValue, Value input) {
    auto buffer = rewriter.create<handshake::BufferOp>(
        rewriter.getUnknownLoc(), type, 1, input, BufferTypeEnum::seq);
    // This does return an unsigned integer but expects signed integers
    // TODO check if this is an MLIR bug
    buffer->setAttr("initValues", rewriter.getI64ArrayAttr({initialValue}));
    return buffer;
  }

public:
  // Binds the provided values to the registers previously created.
  void bindRegInputs(ValueRange inputs) {
    // Connect the back edges
    for (auto [backEdge, input] : llvm::zip(backEdges, inputs))
      backEdge.setValue(input);
  }

  // Builds the registers with the collected regInfo.
  void buildRegisters(SmallVectorImpl<Value> &regs) {
    for (auto regInfo : regInfos) {
      Type regType = regInfo.getType();
      Value input = backEdges.emplace_back(backBuilder.get(regType));
      auto reg = buildRegister(regType, regInfo.getInt(), input);
      regs.push_back(reg);
    }
  }

  size_t getNumRegisters() const { return regInfos.size(); }

private:
  ConversionPatternRewriter &rewriter;
  SmallVector<IntegerAttr> regInfos;
  BackedgeBuilder backBuilder;
  SmallVector<Backedge> backEdges;
};
// Builds a handshake::FuncOp and that represents the mapping funtion. This
// function is then instantiated and connected to its inputs and outputs.
struct MapOpLowering : public StreamOpLowering<MapOp> {
  using StreamOpLowering::StreamOpLowering;

  LogicalResult
  matchAndRewrite(MapOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    TypeConverter *typeConverter = getTypeConverter();

    // create surrounding region
    Region r;

    SmallVector<Type> inputTypes;
    if (failed(typeConverter->convertTypes(op->getOperandTypes(), inputTypes)))
      return failure();

    SmallVector<Location> argLocs(inputTypes.size(), loc);

    Block *entryBlock =
        rewriter.createBlock(&r, r.begin(), inputTypes, argLocs);
    Value tupleIn = entryBlock->getArgument(0);
    Value streamCtrl = entryBlock->getArgument(1);

    auto unpack = rewriter.create<handshake::UnpackOp>(loc, tupleIn);
    Value data = unpack.getResult(0);
    Value eos = unpack.getResult(1);

    Block *lambda = &op.getRegion().front();
    SmallVector<Value> lambdaIns;
    lambdaIns.push_back(data);

    auto regBuilder = RegisterBuilder::create(rewriter, op);
    if (failed(regBuilder))
      return failure();

    regBuilder->buildRegisters(lambdaIns);

    // Ctrl is at the end, due to a handshake limitation
    lambdaIns.push_back(streamCtrl);

    rewriter.setInsertionPointToStart(entryBlock);
    rewriter.mergeBlocks(lambda, entryBlock, lambdaIns);

    Operation *oldTerm = entryBlock->getTerminator();

    regBuilder->bindRegInputs(llvm::drop_begin(oldTerm->getOperands(), 1));

    rewriter.setInsertionPoint(oldTerm);
    auto tupleOut = rewriter.create<handshake::PackOp>(
        oldTerm->getLoc(), ValueRange({oldTerm->getOperand(0), eos}));

    SmallVector<Value> newTermOperands = {tupleOut,
                                          oldTerm->getOperands().back()};
    auto newTerm = rewriter.replaceOpWithNewOp<handshake::ReturnOp>(
        oldTerm, newTermOperands);

    TypeRange resTypes = newTerm->getOperandTypes();

    SmallVector<Value> operands;
    resolveNewOperands(op, adaptor.getOperands(), operands);

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

  LogicalResult
  matchAndRewrite(FilterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    TypeConverter *typeConverter = getTypeConverter();

    Region r;

    SmallVector<Type> inputTypes;
    if (failed(typeConverter->convertTypes(op->getOperandTypes(), inputTypes)))
      return failure();

    SmallVector<Location> argLocs(inputTypes.size(), loc);

    Block *entryBlock =
        rewriter.createBlock(&r, r.begin(), inputTypes, argLocs);
    Value tupleIn = entryBlock->getArgument(0);
    Value streamCtrl = entryBlock->getArgument(1);

    auto unpack = rewriter.create<handshake::UnpackOp>(loc, tupleIn);
    Value data = unpack.getResult(0);
    Value eos = unpack.getResult(1);

    Block *lambda = &op.getRegion().front();
    SmallVector<Value> lambdaIns;
    lambdaIns.push_back(data);

    auto regBuilder = RegisterBuilder::create(rewriter, op);
    if (failed(regBuilder))
      return failure();

    regBuilder->buildRegisters(lambdaIns);

    // Ctrl is at the end, due to a handshake limitation
    lambdaIns.push_back(streamCtrl);
    rewriter.mergeBlocks(lambda, entryBlock, lambdaIns);

    Operation *oldTerm = entryBlock->getTerminator();

    // Connect the back edges
    regBuilder->bindRegInputs(llvm::drop_begin(oldTerm->getOperands(), 1));

    rewriter.setInsertionPointToEnd(entryBlock);

    Value cond = oldTerm->getOperand(0);
    Value ctrl = oldTerm->getOperands().back();

    auto tupleOut =
        rewriter.create<handshake::PackOp>(loc, ValueRange({data, eos}));

    auto condOrEos = rewriter.create<arith::OrIOp>(loc, cond, eos);

    auto dataBr = rewriter.create<handshake::ConditionalBranchOp>(
        rewriter.getUnknownLoc(), condOrEos, tupleOut);

    // Makes sure we only emit Ctrl when data is produced
    auto ctrlBr = rewriter.create<handshake::ConditionalBranchOp>(
        rewriter.getUnknownLoc(), condOrEos, ctrl);

    SmallVector<Value> newTermOperands = {dataBr.getTrueResult(),
                                          ctrlBr.getTrueResult()};
    auto newTerm = rewriter.replaceOpWithNewOp<handshake::ReturnOp>(
        oldTerm, newTermOperands);

    SmallVector<Value> operands;
    resolveNewOperands(op, adaptor.getOperands(), operands);

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

  LogicalResult
  matchAndRewrite(ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    TypeConverter *typeConverter = getTypeConverter();
    SmallVector<Type> resultTypes;
    if (failed(
            typeConverter->convertType(op.getResult().getType(), resultTypes)))
      return failure();

    assert(resultTypes[0].isa<TupleType>());
    Type resultType = resultTypes[0].dyn_cast<TupleType>().getType(0);

    // TODO: handshake currently only supports i64 buffers, change this as
    // soon as support for other types is added.
    assert(resultType == rewriter.getI64Type() &&
           "currently, only i64 buffers are supported");

    Region r;

    SmallVector<Type> inputTypes;
    if (failed(typeConverter->convertTypes(op->getOperandTypes(), inputTypes)))
      return failure();

    SmallVector<Location> argLocs(inputTypes.size(), loc);

    Block *entryBlock =
        rewriter.createBlock(&r, r.begin(), inputTypes, argLocs);
    Value tupleIn = entryBlock->getArgument(0);
    Value streamCtrl = entryBlock->getArgument(1);

    auto unpack = rewriter.create<handshake::UnpackOp>(loc, tupleIn);
    Value data = unpack.getResult(0);
    Value eos = unpack.getResult(1);

    Block *lambda = &op.getRegion().front();

    Operation *oldTerm = lambda->getTerminator();
    auto buffer = rewriter.create<handshake::BufferOp>(
        rewriter.getUnknownLoc(), resultType, 1, oldTerm->getOperand(0),
        BufferTypeEnum::seq);
    // This does return an unsigned integer but expects signed integers
    // TODO check if this is an MLIR bug
    buffer->setAttr("initValues", rewriter.getI64ArrayAttr(
                                      {(int64_t)adaptor.getInitValue()}));

    auto dataBr = rewriter.create<handshake::ConditionalBranchOp>(
        rewriter.getUnknownLoc(), eos, buffer);
    auto eosBr = rewriter.create<handshake::ConditionalBranchOp>(
        rewriter.getUnknownLoc(), eos, eos);
    auto ctrlBr = rewriter.create<handshake::ConditionalBranchOp>(
        rewriter.getUnknownLoc(), eos, oldTerm->getOperands().back());

    SmallVector<Value> lambdaIns = {dataBr.getFalseResult(), data};

    auto regBuilder = RegisterBuilder::create(rewriter, op);
    if (failed(regBuilder))
      return failure();

    regBuilder->buildRegisters(lambdaIns);

    lambdaIns.push_back(streamCtrl);
    rewriter.mergeBlocks(lambda, entryBlock, lambdaIns);

    // Connect the back edges
    regBuilder->bindRegInputs(llvm::drop_begin(oldTerm->getOperands(), 1));

    rewriter.setInsertionPoint(oldTerm);

    // Connect outputs and ensure correct delay between value and EOS=true
    // emission A sequental buffer ensures a cycle delay of 1
    auto eosFalse = rewriter.create<handshake::ConstantOp>(
        rewriter.getUnknownLoc(),
        rewriter.getIntegerAttr(rewriter.getI1Type(), 0),
        ctrlBr.getTrueResult());
    auto tupleOutVal = rewriter.create<handshake::PackOp>(
        loc, ValueRange({dataBr.getTrueResult(), eosFalse}));

    auto tupleOutEOS = rewriter.create<handshake::PackOp>(
        loc, ValueRange({dataBr.getTrueResult(), eosBr.getTrueResult()}));

    // Not really needed, but the BufferOp builder requires an input
    auto bubble = rewriter.create<ConstantOp>(
        loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0),
        ctrlBr.getTrueResult());
    auto select = rewriter.create<handshake::BufferOp>(
        rewriter.getUnknownLoc(), rewriter.getI32Type(), 2, bubble,
        BufferTypeEnum::seq);
    // First select the tupleOut, afterwards the one with the EOS signal
    select->setAttr("initValues", rewriter.getI64ArrayAttr({1, 0}));

    auto tupleOut = rewriter.create<MuxOp>(
        loc, select, ValueRange({tupleOutVal, tupleOutEOS}));

    auto eosCtrl = rewriter.create<JoinOp>(loc, tupleOutEOS.getResult());
    auto ctrlOut = rewriter.create<MuxOp>(
        loc, select, ValueRange({ctrlBr.getTrueResult(), eosCtrl}));

    SmallVector<Value> newTermOperands = {tupleOut, ctrlOut};

    auto newTerm = rewriter.replaceOpWithNewOp<handshake::ReturnOp>(
        oldTerm, newTermOperands);

    SmallVector<Value> operands;
    resolveNewOperands(op, adaptor.getOperands(), operands);

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

  LogicalResult
  matchAndRewrite(stream::PackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<handshake::PackOp>(op, adaptor.getOperands());

    return success();
  }
};

struct UnpackOpLowering : public OpConversionPattern<stream::UnpackOp> {
  using OpConversionPattern<stream::UnpackOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(stream::UnpackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<handshake::UnpackOp>(op, adaptor.getInput());

    return success();
  }
};

struct SplitOpLowering : public StreamOpLowering<SplitOp> {
  using StreamOpLowering::StreamOpLowering;

  LogicalResult
  matchAndRewrite(SplitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    TypeConverter *typeConverter = getTypeConverter();

    // create surrounding region
    Region r;

    SmallVector<Type> inputTypes;
    if (failed(typeConverter->convertTypes(op->getOperandTypes(), inputTypes)))
      return failure();

    SmallVector<Location> argLocs(inputTypes.size(), loc);

    Block *entryBlock =
        rewriter.createBlock(&r, r.begin(), inputTypes, argLocs);
    Value tupleIn = entryBlock->getArgument(0);
    Value streamCtrl = entryBlock->getArgument(1);

    auto unpack = rewriter.create<handshake::UnpackOp>(loc, tupleIn);
    Value data = unpack.getResult(0);
    Value eos = unpack.getResult(1);

    Block *lambda = &op.getRegion().front();
    SmallVector<Value> lambdaIns = {data};

    auto regBuilder = RegisterBuilder::create(rewriter, op);
    if (failed(regBuilder))
      return failure();

    regBuilder->buildRegisters(lambdaIns);

    lambdaIns.push_back(streamCtrl);
    rewriter.mergeBlocks(lambda, entryBlock, lambdaIns);

    Operation *oldTerm = entryBlock->getTerminator();
    size_t numRegs = regBuilder->getNumRegisters();
    size_t numOuts = oldTerm->getNumOperands() - 1 - numRegs;

    // Connect the back edges
    regBuilder->bindRegInputs(
        llvm::drop_begin(oldTerm->getOperands(), numOuts));

    rewriter.setInsertionPoint(oldTerm);

    SmallVector<Value> newTermOperands;
    for (auto oldOp : llvm::drop_end(oldTerm->getOperands(), numRegs + 1)) {
      auto pack = rewriter.create<handshake::PackOp>(oldTerm->getLoc(),
                                                     ValueRange({oldOp, eos}));
      newTermOperands.push_back(pack.getResult());
      newTermOperands.push_back(oldTerm->getOperands().back());
    }

    auto newTerm = rewriter.replaceOpWithNewOp<handshake::ReturnOp>(
        oldTerm, newTermOperands);

    TypeRange resTypes = newTerm->getOperandTypes();

    SmallVector<Value> operands;
    resolveNewOperands(op, adaptor.getOperands(), operands);

    rewriter.setInsertionPointToStart(getTopLevelBlock(op));
    FuncOp newFuncOp =
        createFuncOp(r, symbolUniquer.getUniqueSymName(op),
                     entryBlock->getArgumentTypes(), resTypes, rewriter);

    replaceWithInstance(op, newFuncOp, operands, rewriter);

    return success();
  }
};

/// TODO: make this more efficient
template <typename Op>
static Value buildReduceTree(ValueRange values, Location loc,
                             ConversionPatternRewriter &rewriter) {
  assert(values.size() > 0);
  Value res = values.front();
  for (auto val : values.drop_front()) {
    res = rewriter.create<Op>(loc, res, val);
  }
  return res;
}

struct CombineOpLowering : public StreamOpLowering<CombineOp> {
  using StreamOpLowering::StreamOpLowering;

  LogicalResult
  matchAndRewrite(CombineOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    TypeConverter *typeConverter = getTypeConverter();

    // create surrounding region
    Region r;

    SmallVector<Type> inputTypes;
    if (failed(typeConverter->convertTypes(op->getOperandTypes(), inputTypes)))
      return failure();

    SmallVector<Location> argLocs(inputTypes.size(), loc);

    Block *entryBlock =
        rewriter.createBlock(&r, r.begin(), inputTypes, argLocs);

    SmallVector<Value> blockInputs;
    SmallVector<Value> eosInputs;
    SmallVector<Value> ctrlInputs;
    for (unsigned i = 0, e = entryBlock->getNumArguments() - 1; i < e; i += 2) {
      Value tupleIn = entryBlock->getArgument(i);
      Value streamCtrl = entryBlock->getArgument(i + 1);
      auto unpack = rewriter.create<handshake::UnpackOp>(loc, tupleIn);
      Value data = unpack.getResult(0);
      Value eos = unpack.getResult(1);

      blockInputs.push_back(data);
      ctrlInputs.push_back(streamCtrl);
      eosInputs.push_back(eos);
    }
    Block *lambda = &op.getRegion().front();

    auto regBuilder = RegisterBuilder::create(rewriter, op);
    if (failed(regBuilder))
      return failure();

    regBuilder->buildRegisters(blockInputs);

    // only execute region when ALL inputs are ready
    auto ctrlJoin = rewriter.create<JoinOp>(loc, ctrlInputs);
    blockInputs.push_back(ctrlJoin);

    rewriter.mergeBlocks(lambda, entryBlock, blockInputs);

    Operation *oldTerm = entryBlock->getTerminator();
    size_t numRegs = regBuilder->getNumRegisters();
    size_t numOuts = oldTerm->getNumOperands() - 1 - numRegs;

    // Connect the back edges
    regBuilder->bindRegInputs(
        llvm::drop_begin(oldTerm->getOperands(), numOuts));
    rewriter.setInsertionPoint(oldTerm);

    // TODO What to do when not all streams provide an eos signal
    Value eos = buildReduceTree<arith::OrIOp>(eosInputs, loc, rewriter);

    SmallVector<Value> newTermOperands;
    for (auto oldOp : oldTerm->getOperands().drop_back()) {
      auto pack = rewriter.create<handshake::PackOp>(oldTerm->getLoc(),
                                                     ValueRange({oldOp, eos}));
      newTermOperands.push_back(pack.getResult());
      newTermOperands.push_back(oldTerm->getOperands().back());
    }

    auto newTerm = rewriter.replaceOpWithNewOp<handshake::ReturnOp>(
        oldTerm, newTermOperands);

    TypeRange resTypes = newTerm->getOperandTypes();

    SmallVector<Value> operands;
    resolveNewOperands(op, adaptor.getOperands(), operands);

    rewriter.setInsertionPointToStart(getTopLevelBlock(op));
    FuncOp newFuncOp =
        createFuncOp(r, symbolUniquer.getUniqueSymName(op),
                     entryBlock->getArgumentTypes(), resTypes, rewriter);

    replaceWithInstance(op, newFuncOp, operands, rewriter);

    return success();
  }
};

struct SinkOpLowering : public StreamOpLowering<stream::SinkOp> {
  using StreamOpLowering::StreamOpLowering;

  LogicalResult
  matchAndRewrite(stream::SinkOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value> operands;
    resolveNewOperands(op, adaptor.getOperands(), operands);

    // Inserts sinks for all inputs
    for (auto operand : operands)
      rewriter.create<handshake::SinkOp>(loc, operand);

    rewriter.eraseOp(op);

    return success();
  }
};

static void
populateStreamToHandshakePatterns(StreamTypeConverter &typeConverter,
                                  SymbolUniquer symbolUniquer,
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
    SplitOpLowering,
    CombineOpLowering,
    SinkOpLowering
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

/// Removes all forks and syncs as the insertion is not able to extend existing
/// forks
static LogicalResult dematerializeForksAndSinks(Region &r) {
  for (auto sinkOp : llvm::make_early_inc_range(r.getOps<handshake::SinkOp>()))
    sinkOp.erase();

  for (auto forkOp :
       llvm::make_early_inc_range(r.getOps<handshake::ForkOp>())) {
    for (auto res : forkOp->getResults())
      res.replaceAllUsesWith(forkOp.getOperand());
    forkOp.erase();
  }
  return success();
}

// TODO Do this with an op trait?
bool isStreamOp(Operation *op) {
  return isa<MapOp, FilterOp, ReduceOp, SplitOp, CombineOp>(op);
}

/// Traverses the modules region recursively and applies the std to handshake
/// conversion on each stream operation region.
LogicalResult transformStdRegions(ModuleOp m) {
  // go over all stream ops and transform their regions
  for (auto funcOp : llvm::make_early_inc_range(m.getOps<func::FuncOp>())) {
    if (funcOp.isDeclaration())
      continue;
    Region *funcRegion = funcOp.getCallableRegion();
    for (Operation &op : funcRegion->getOps()) {
      if (!isStreamOp(&op))
        continue;
      for (auto &r : op.getRegions()) {
        StreamLowering sl(r);
        if (failed(lowerRegion<YieldOp>(sl, false, false)))
          return failure();
        if (failed(dematerializeForksAndSinks(r)))
          return failure();
        removeBasicBlocks(r);
      }
    }
  }
  return success();
}

static LogicalResult removeUnusedConversionCasts(ModuleOp m) {
  for (auto funcOp : m.getOps<handshake::FuncOp>()) {
    if (funcOp.isDeclaration())
      continue;
    Region &funcRegion = funcOp.getBody();
    for (auto op : llvm::make_early_inc_range(
             funcRegion.getOps<UnrealizedConversionCastOp>())) {
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

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    if (failed(removeUnusedConversionCasts(getOperation()))) {
      signalPassFailure();
      return;
    }

    if (failed(materializeForksAndSinks(getOperation()))) {
      signalPassFailure();
      return;
    }

    if (failed(postDataflowConvert(getOperation())))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> circt_stream::createStreamToHandshakePass() {
  return std::make_unique<StreamToHandshakePass>();
}

