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

#include "circt-stream/Dialect/Stream/StreamDialect.h"
#include "circt-stream/Dialect/Stream/StreamOps.h"
#include "circt/Conversion/StandardToHandshake.h"
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/SymCache.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

namespace circt_stream {
#define GEN_PASS_DEF_STREAMTOHANDSHAKE
#include "circt-stream/Conversion/Passes.h.inc"
} // namespace circt_stream

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

    addConversion([](StreamType type) {
      MLIRContext *ctx = type.getContext();
      return TupleType::get(ctx,
                            {type.getElementType(), IntegerType::get(ctx, 1)});
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

struct ReturnOpLowering : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<handshake::ReturnOp>(op, adaptor.getOperands());
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
  return rewriter.replaceOpWithNewOp<InstanceOp>(op, func, newOperands);
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
  /// TODO: No need to be generic, just pass in the registers
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
  RegisterBuilder(ConversionPatternRewriter &rewriter, Value eos, Location loc)
      : rewriter(rewriter), backBuilder(rewriter, rewriter.getUnknownLoc()),
        eos(eos), loc(loc) {}

public:
  template <typename TOp>
  static FailureOr<RegisterBuilder> create(ConversionPatternRewriter &rewriter,
                                           TOp op, Value eos) {
    RegisterBuilder regBuilder(rewriter, eos, op.getLoc());
    if (failed(getRegInfos(op, regBuilder.regInfos)))
      return failure();

    return regBuilder;
  }

private:
  // TODO initialValue type: handshake buffers do currently only support
  // integers
  Value buildRegister(Type type, int64_t initialValue, Value input) {
    auto source = rewriter.create<handshake::SourceOp>(loc);
    auto initValConst = rewriter.create<handshake::ConstantOp>(
        loc, rewriter.getIntegerAttr(type, initialValue), source);
    auto buffInSel = rewriter.create<handshake::MuxOp>(
        loc, eos, ValueRange({input, initValConst}));

    auto buffer = rewriter.create<handshake::BufferOp>(
        rewriter.getUnknownLoc(), type, 1, buffInSel, BufferTypeEnum::seq);
    // This does return an unsigned integer but expects signed integers
    // TODO check if this is an MLIR bug
    buffer->setAttr("initValues", rewriter.getI64ArrayAttr({initialValue}));

    // Only emit a result on eos == 0
    auto dataBr =
        rewriter.create<handshake::ConditionalBranchOp>(loc, eos, buffer);
    return dataBr.getFalseResult();
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
  Value eos;
  Location loc;
};

static Value getDefaultVal(Location loc, Type type, Value trigger,
                           ConversionPatternRewriter &rewriter) {
  if (type.isa<NoneType>())
    return trigger;

  if (!type.isa<TupleType>())
    return rewriter.create<ConstantOp>(loc, rewriter.getIntegerAttr(type, 0),
                                       trigger);

  auto tupleType = type.dyn_cast<TupleType>();
  SmallVector<Value> packInputs;
  for (auto elementType : tupleType.getTypes())
    packInputs.push_back(getDefaultVal(loc, elementType, trigger, rewriter));

  return rewriter.create<handshake::PackOp>(loc, packInputs);
}

static Value dataOrDefault(Location loc, Value data, Value eos,
                           ConversionPatternRewriter &rewriter) {
  // Only output the lambdas result when it will produce one. On EOS, the
  // lambda is bypassed, so we select the input value.
  auto source = rewriter.create<SourceOp>(loc);
  return rewriter.create<MuxOp>(
      loc, eos,
      ValueRange({data, getDefaultVal(loc, data.getType(), source, rewriter)}));
}

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

    auto unpack = rewriter.create<handshake::UnpackOp>(loc, tupleIn);
    Value data = unpack.getResult(0);
    Value eos = unpack.getResult(1);

    // Only feed data into circuit on eos == 0
    auto dataBr =
        rewriter.create<handshake::ConditionalBranchOp>(loc, eos, data);

    Block *lambda = &op.getRegion().front();
    SmallVector<Value> lambdaIns;
    lambdaIns.push_back(dataBr.getFalseResult());

    auto regBuilder = RegisterBuilder::create(rewriter, op, eos);
    if (failed(regBuilder))
      return failure();

    regBuilder->buildRegisters(lambdaIns);

    // Ctrl is at the end, due to a handshake limitation
    Value lambdaCtrl = rewriter.create<JoinOp>(loc, dataBr.getFalseResult());
    lambdaIns.push_back(lambdaCtrl);

    rewriter.setInsertionPointToStart(entryBlock);
    rewriter.mergeBlocks(lambda, entryBlock, lambdaIns);

    Operation *oldTerm = entryBlock->getTerminator();

    regBuilder->bindRegInputs(llvm::drop_begin(oldTerm->getOperands(), 1));

    // Only output the lambdas result when it will produce one. On EOS, the
    // lambda is bypassed, so we select the input value.
    auto outData = dataOrDefault(loc, oldTerm->getOperand(0), eos, rewriter);
    // Note: we ignore the outgoing control signal.

    rewriter.setInsertionPoint(oldTerm);
    auto tupleOut = rewriter.create<handshake::PackOp>(
        oldTerm->getLoc(), ValueRange({outData, eos}));

    auto newTerm = rewriter.replaceOpWithNewOp<handshake::ReturnOp>(
        oldTerm, tupleOut.getResult());

    TypeRange resTypes = newTerm->getOperandTypes();

    rewriter.setInsertionPointToStart(getTopLevelBlock(op));
    FuncOp newFuncOp =
        createFuncOp(r, symbolUniquer.getUniqueSymName(op),
                     entryBlock->getArgumentTypes(), resTypes, rewriter);

    replaceWithInstance(op, newFuncOp, adaptor.getOperands(), rewriter);

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

    auto unpack = rewriter.create<handshake::UnpackOp>(loc, tupleIn);
    Value data = unpack.getResult(0);
    Value eos = unpack.getResult(1);

    // Only feed data into circuit on eos == 0
    auto inDataBr =
        rewriter.create<handshake::ConditionalBranchOp>(loc, eos, data);

    Block *lambda = &op.getRegion().front();
    SmallVector<Value> lambdaIns;
    lambdaIns.push_back(inDataBr.getFalseResult());

    auto regBuilder = RegisterBuilder::create(rewriter, op, eos);
    if (failed(regBuilder))
      return failure();

    regBuilder->buildRegisters(lambdaIns);

    // Ctrl is at the end, due to a handshake limitation
    Value lambdaCtrl = rewriter.create<JoinOp>(loc, inDataBr.getFalseResult());
    lambdaIns.push_back(lambdaCtrl);
    rewriter.mergeBlocks(lambda, entryBlock, lambdaIns);

    Operation *oldTerm = entryBlock->getTerminator();

    // Connect the back edges
    regBuilder->bindRegInputs(llvm::drop_begin(oldTerm->getOperands(), 1));

    rewriter.setInsertionPointToEnd(entryBlock);

    Value cond = oldTerm->getOperand(0);

    // Drop eos on the path to "shouldEmit" if it will not be consumed
    auto eosBr = rewriter.create<handshake::ConditionalBranchOp>(
        rewriter.getUnknownLoc(), eos, eos);
    // Need a mux here, as the input cond will not be fired on EOS == 1
    auto shouldEmit = rewriter.create<MuxOp>(
        loc, eos, ValueRange({cond, eosBr.getTrueResult()}));

    auto dataBr = rewriter.create<handshake::ConditionalBranchOp>(
        rewriter.getUnknownLoc(), shouldEmit, tupleIn);

    auto newTerm = rewriter.replaceOpWithNewOp<handshake::ReturnOp>(
        oldTerm, dataBr.getTrueResult());

    rewriter.setInsertionPointToStart(getTopLevelBlock(op));
    FuncOp newFuncOp = createFuncOp(r, symbolUniquer.getUniqueSymName(op),
                                    entryBlock->getArgumentTypes(),
                                    newTerm.getOperandTypes(), rewriter);
    replaceWithInstance(op, newFuncOp, adaptor.getOperands(), rewriter);
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
    BackedgeBuilder backBuilder(rewriter, loc);

    // TODO add a helper for that?
    assert(resultTypes[0].isa<TupleType>());
    Type resultType = resultTypes[0].dyn_cast<TupleType>().getType(0);

    Region r;

    SmallVector<Type> inputTypes;
    if (failed(typeConverter->convertTypes(op->getOperandTypes(), inputTypes)))
      return failure();

    SmallVector<Location> argLocs(inputTypes.size(), loc);

    Block *entryBlock =
        rewriter.createBlock(&r, r.begin(), inputTypes, argLocs);
    Value tupleIn = entryBlock->getArgument(0);

    auto unpack = rewriter.create<handshake::UnpackOp>(loc, tupleIn);
    Value data = unpack.getResult(0);
    Value eos = unpack.getResult(1);

    Block *lambda = &op.getRegion().front();

    Operation *oldTerm = lambda->getTerminator();

    // Buffer which functions as a register
    // Select the init value to be feed in the buffer when EOS was asserted
    // to corretly reinitialize the register.
    auto source = rewriter.create<handshake::SourceOp>(loc);
    auto initValConst = rewriter.create<handshake::ConstantOp>(
        loc, rewriter.getIntegerAttr(resultType, adaptor.getInitValue()),
        source);
    auto buffInSel = rewriter.create<handshake::MuxOp>(
        loc, eos, ValueRange({oldTerm->getOperand(0), initValConst}));
    auto buffer = rewriter.create<handshake::BufferOp>(
        rewriter.getUnknownLoc(), resultType, 1,
        buffInSel, // oldTerm->getOperand(0),
        BufferTypeEnum::seq);

    // This does return an unsigned integer but expects signed integers
    // TODO check if this is an MLIR bug
    buffer->setAttr("initValues", rewriter.getI64ArrayAttr(
                                      {(int64_t)adaptor.getInitValue()}));

    auto dataBr = rewriter.create<handshake::ConditionalBranchOp>(
        rewriter.getUnknownLoc(), eos, buffer);
    auto eosBr = rewriter.create<handshake::ConditionalBranchOp>(
        rewriter.getUnknownLoc(), eos, eos);

    // Only feed data in circuit when eos is not asserted
    auto inDataBr = rewriter.create<handshake::ConditionalBranchOp>(
        rewriter.getUnknownLoc(), eos, data);
    SmallVector<Value> lambdaIns = {dataBr.getFalseResult(),
                                    inDataBr.getFalseResult()};

    auto regBuilder = RegisterBuilder::create(rewriter, op, eos);
    if (failed(regBuilder))
      return failure();

    regBuilder->buildRegisters(lambdaIns);

    Value lambdaCtrl = rewriter.create<JoinOp>(loc, dataBr.getFalseResult());
    lambdaIns.push_back(lambdaCtrl);
    rewriter.mergeBlocks(lambda, entryBlock, lambdaIns);

    // Connect the back edges
    regBuilder->bindRegInputs(llvm::drop_begin(oldTerm->getOperands(), 1));

    rewriter.setInsertionPoint(oldTerm);

    // Connect outputs and ensure correct delay between value and EOS=true
    // emission A sequental buffer ensures a cycle delay of 1
    auto trigger = rewriter.create<JoinOp>(loc, dataBr.getTrueResult());
    auto eosFalse = rewriter.create<handshake::ConstantOp>(
        rewriter.getUnknownLoc(),
        rewriter.getIntegerAttr(rewriter.getI1Type(), 0), trigger);
    auto tupleOutVal = rewriter.create<handshake::PackOp>(
        loc, ValueRange({dataBr.getTrueResult(), eosFalse}));

    auto tupleOutEOS = rewriter.create<handshake::PackOp>(
        loc, ValueRange({dataBr.getTrueResult(), eosBr.getTrueResult()}));

    // The buffer must cycle to itself, such that it can be reused again.
    auto bufferCycle = backBuilder.get(rewriter.getI1Type());
    auto select = rewriter.create<handshake::BufferOp>(
        rewriter.getUnknownLoc(), rewriter.getI1Type(), 2, bufferCycle,
        BufferTypeEnum::seq);
    // First select the tupleOut, afterwards the one with the EOS signal
    select->setAttr("initValues", rewriter.getI64ArrayAttr({1, 0}));

    bufferCycle.setValue(select);

    auto tupleOut = rewriter.create<MuxOp>(
        loc, select, ValueRange({tupleOutVal, tupleOutEOS}));

    auto newTerm = rewriter.replaceOpWithNewOp<handshake::ReturnOp>(
        oldTerm, tupleOut.getResult());

    rewriter.setInsertionPointToStart(getTopLevelBlock(op));
    FuncOp newFuncOp = createFuncOp(r, symbolUniquer.getUniqueSymName(op),
                                    entryBlock->getArgumentTypes(),
                                    newTerm.getOperandTypes(), rewriter);

    replaceWithInstance(op, newFuncOp, adaptor.getOperands(), rewriter);
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

    auto unpack = rewriter.create<handshake::UnpackOp>(loc, tupleIn);
    Value data = unpack.getResult(0);
    Value eos = unpack.getResult(1);

    // Only feed data into circuit on eos == 0
    auto dataBr =
        rewriter.create<handshake::ConditionalBranchOp>(loc, eos, data);

    Block *lambda = &op.getRegion().front();
    SmallVector<Value> lambdaIns = {dataBr.getFalseResult()};

    auto regBuilder = RegisterBuilder::create(rewriter, op, eos);
    if (failed(regBuilder))
      return failure();

    regBuilder->buildRegisters(lambdaIns);

    Value lambdaCtrl = rewriter.create<JoinOp>(loc, dataBr.getFalseResult());
    lambdaIns.push_back(lambdaCtrl);
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
      // Only output the lambdas result when it will produce one. On EOS, the
      // lambda is bypassed, so we select the input value.
      auto outData = dataOrDefault(loc, oldOp, eos, rewriter);
      auto pack = rewriter.create<handshake::PackOp>(
          oldTerm->getLoc(), ValueRange({outData, eos}));
      newTermOperands.push_back(pack.getResult());
    }

    auto newTerm = rewriter.replaceOpWithNewOp<handshake::ReturnOp>(
        oldTerm, newTermOperands);

    TypeRange resTypes = newTerm->getOperandTypes();

    rewriter.setInsertionPointToStart(getTopLevelBlock(op));
    FuncOp newFuncOp =
        createFuncOp(r, symbolUniquer.getUniqueSymName(op),
                     entryBlock->getArgumentTypes(), resTypes, rewriter);

    replaceWithInstance(op, newFuncOp, adaptor.getOperands(), rewriter);

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

    SmallVector<Value> dataInputs;
    SmallVector<Value> eosInputs;
    for (Value tupleIn : entryBlock->getArguments()) {
      auto unpack = rewriter.create<handshake::UnpackOp>(loc, tupleIn);
      Value data = unpack.getResult(0);
      Value eos = unpack.getResult(1);

      dataInputs.push_back(data);
      eosInputs.push_back(eos);
    }
    Block *lambda = &op.getRegion().front();

    // TODO What to do when not all streams provide an eos signal
    Value eos = buildReduceTree<arith::OrIOp>(eosInputs, loc, rewriter);

    // Only feed data into circuit on eos == 0
    SmallVector<Value> blockInputs;
    for (auto data : dataInputs)
      blockInputs.push_back(
          rewriter.create<handshake::ConditionalBranchOp>(loc, eos, data)
              .getFalseResult());

    // Trigger function if all inputs are here
    auto ctrlJoin = rewriter.create<JoinOp>(loc, blockInputs);

    auto regBuilder = RegisterBuilder::create(rewriter, op, eos);
    if (failed(regBuilder))
      return failure();

    regBuilder->buildRegisters(blockInputs);

    // only execute region when ALL inputs are ready
    blockInputs.push_back(ctrlJoin);

    rewriter.mergeBlocks(lambda, entryBlock, blockInputs);

    Operation *oldTerm = entryBlock->getTerminator();
    size_t numRegs = regBuilder->getNumRegisters();
    size_t numOuts = oldTerm->getNumOperands() - 1 - numRegs;

    // Connect the back edges
    regBuilder->bindRegInputs(
        llvm::drop_begin(oldTerm->getOperands(), numOuts));
    rewriter.setInsertionPoint(oldTerm);

    auto outData = dataOrDefault(loc, oldTerm->getOperand(0), eos, rewriter);
    auto pack =
        rewriter.create<handshake::PackOp>(loc, ValueRange({outData, eos}));

    auto newTerm = rewriter.replaceOpWithNewOp<handshake::ReturnOp>(
        oldTerm, pack.getResult());

    TypeRange resTypes = newTerm->getOperandTypes();

    rewriter.setInsertionPointToStart(getTopLevelBlock(op));
    FuncOp newFuncOp =
        createFuncOp(r, symbolUniquer.getUniqueSymName(op),
                     entryBlock->getArgumentTypes(), resTypes, rewriter);

    replaceWithInstance(op, newFuncOp, adaptor.getOperands(), rewriter);

    return success();
  }
};

struct SinkOpLowering : public StreamOpLowering<stream::SinkOp> {
  using StreamOpLowering::StreamOpLowering;

  LogicalResult
  matchAndRewrite(stream::SinkOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Inserts sinks for all inputs
    for (auto operand : adaptor.getOperands())
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
    : public circt_stream::impl::StreamToHandshakeBase<StreamToHandshakePass> {
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
    target.addLegalDialect<arith::ArithDialect>();
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

