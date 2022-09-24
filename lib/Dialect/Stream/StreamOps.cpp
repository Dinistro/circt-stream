//===- StreamOps.cpp - Stream dialect ops -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-stream/Dialect/Stream/StreamOps.h"

#include <llvm/ADT/STLExtras.h>

#include "circt-stream/Dialect/Stream/StreamDialect.h"
#include "circt-stream/Dialect/Stream/StreamTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt_stream;
using namespace circt_stream::stream;

#define GET_OP_CLASSES
#include "circt-stream/Dialect/Stream/StreamOps.cpp.inc"

static LogicalResult verifyRegionArgs(Operation *op, TypeRange expectedTypes,
                                      Region &r) {
  if (r.getNumArguments() != expectedTypes.size())
    return op->emitError("expect region to have ")
           << expectedTypes.size() << " arguments.";

  unsigned i = 0;
  for (auto [expected, actual] :
       llvm::zip(expectedTypes, r.getArgumentTypes())) {
    if (expected != actual)
      return op->emitError("expect the block argument #")
             << i << " to have type " << expected << ", got " << actual
             << " instead.";
    i++;
  }

  return success();
}

static LogicalResult verifyYieldOperands(Operation *op, TypeRange returnTypes,
                                         Region &r) {
  for (auto term : r.getOps<YieldOp>()) {
    if (term.getNumOperands() != returnTypes.size())
      return term.emitError("expect ")
             << returnTypes.size() << " operands, got "
             << term.getNumOperands();
    unsigned i = 0;
    for (auto [expected, actual] :
         llvm::zip(returnTypes, term.getOperandTypes())) {
      if (expected != actual)
        return term.emitError("expect the operand #")
               << i << " to have type " << expected << ", got " << actual
               << " instead.";
      i++;
    }
  }
  return success();
}

static Type getElementType(Type streamType) {
  assert(streamType.isa<StreamType>() &&
         "can only extract element type of a StreamType");
  return streamType.cast<StreamType>().getElementType();
}

/// Verifies that a region has indeed the expected inputs and that all
/// terminators return operands matching the provided return types.
static LogicalResult verifyRegion(Operation *op, Region &r,
                                  TypeRange inputTypes, TypeRange returnTypes) {
  // Check arguments
  if (failed(verifyRegionArgs(op, inputTypes, r)))
    return failure();

  // Check terminator
  if (failed(verifyYieldOperands(op, returnTypes, r)))
    return failure();

  return success();
}

/// Verifies that a region can handle the input streams and always returns
/// elements with the type of the output stream.
static LogicalResult verifyDefaultRegion(Operation *op, Region &r,
                                         TypeRange registerTypes = {}) {
  // TODO where to append them?
  SmallVector<Type> inputTypes =
      llvm::to_vector(llvm::map_range(op->getOperandTypes(), getElementType));
  llvm::copy(registerTypes, std::back_inserter(inputTypes));

  SmallVector<Type> returnTypes =
      llvm::to_vector(llvm::map_range(op->getResultTypes(), getElementType));
  llvm::copy(registerTypes, std::back_inserter(returnTypes));

  return verifyRegion(op, r, inputTypes, returnTypes);
}

template <typename TOp>
static LogicalResult getRegTypes(TOp op, SmallVectorImpl<Type> &regTypes) {
  Optional<ArrayAttr> regTypeAttr = op.getRegisters();
  if (regTypeAttr.has_value()) {
    for (auto attr : *regTypeAttr) {
      auto res = llvm::TypeSwitch<Attribute, LogicalResult>(attr)
                     .Case<IntegerAttr>([&](auto intAttr) {
                       regTypes.push_back(intAttr.getType());
                       return success();
                     })
                     .Default([&](Attribute) {
                       return op->emitError("unsupported register type");
                     });
      if (failed(res))
        return failure();
    }
  }
  return success();
}

LogicalResult MapOp::verifyRegions() {
  SmallVector<Type> regTypes;
  if (failed(getRegTypes(*this, regTypes)))
    return failure();

  return verifyDefaultRegion(getOperation(), getRegion(), regTypes);
}

LogicalResult FilterOp::verifyRegions() {
  SmallVector<Type> inputTypes = llvm::to_vector(
      llvm::map_range((*this)->getOperandTypes(), getElementType));

  SmallVector<Type> regTypes;
  if (failed(getRegTypes(*this, regTypes)))
    return failure();
  llvm::copy(regTypes, std::back_inserter(inputTypes));

  SmallVector<Type> resultTypes;
  resultTypes.push_back(IntegerType::get(this->getContext(), 1));
  llvm::copy(regTypes, std::back_inserter(resultTypes));

  return verifyRegion(getOperation(), getRegion(), inputTypes, resultTypes);
}

LogicalResult ReduceOp::verifyRegions() {
  SmallVector<Type> regTypes;
  if (failed(getRegTypes(*this, regTypes)))
    return failure();

  Type inputType = getElementType(getInput().getType());
  Type accType = getElementType(getResult().getType());

  SmallVector<Type> inputTypes = {accType, inputType};
  llvm::copy(regTypes, std::back_inserter(inputTypes));

  SmallVector<Type> resultTypes = {accType};
  llvm::copy(regTypes, std::back_inserter(resultTypes));

  return verifyRegion(getOperation(), getRegion(), inputTypes, resultTypes);
}

ParseResult UnpackOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand tuple;
  TupleType type;

  if (parser.parseOperand(tuple) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(type))
    return failure();

  if (parser.resolveOperand(tuple, type, result.operands))
    return failure();

  result.addTypes(type.getTypes());

  return success();
}

void UnpackOp::print(OpAsmPrinter &p) {
  p << " " << getInput();
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << getInput().getType();
}

/// Replaces unnecessary `stream.unpack` when its operand is the result of a
/// `stream.pack` operation. In the following snippet, all uses of `%a2` and
/// `%b2` are replaced with `%a` and `%b` respectively.
///
/// ```
///   %tuple = stream.pack %a, %b {attributes} : tuple<i32, i64>
///   %a2, %b2 = stream.unpack %tuple {attributes} : tuple<i32, i64>
///   // ... some ops using %a2, %b2
/// ```
LogicalResult UnpackOp::canonicalize(UnpackOp op, PatternRewriter &rewriter) {
  PackOp tuple = dyn_cast_or_null<PackOp>(op.getInput().getDefiningOp());
  if (!tuple)
    return failure();

  rewriter.replaceOp(op, tuple.getInputs());
  return success();
}

ParseResult PackOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  TupleType type;

  if (parser.parseOperandList(operands) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(type))
    return failure();

  if (parser.resolveOperands(operands, type.getTypes(), allOperandLoc,
                             result.operands))
    return failure();

  result.addTypes(type);

  return success();
}

void PackOp::print(OpAsmPrinter &p) {
  p << " " << getInputs();
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << getResult().getType();
}

/// Replaces an unnecessary `stream.pack` when it's operands are results of a
/// `stream.unpack` op. In the following snippet, all uses of `%tuple2` are
/// replaced with `%tuple`.
///
/// ```
///   %a, %b = stream.unpack %tuple {attributes} : tuple<i32, i64>
///   %tuple2 = stream.pack %a, %b {attributes} : tuple<i32, i64>
///   // ... some ops using %tuple2
/// ```
LogicalResult PackOp::canonicalize(PackOp op, PatternRewriter &rewriter) {
  if (op.getInputs().size() == 0)
    return failure();

  Operation *singleDefiningOp = op.getInputs()[0].getDefiningOp();

  if (!llvm::all_of(op.getInputs(), [singleDefiningOp](Value input) {
        return input.getDefiningOp() == singleDefiningOp;
      }))
    return failure();

  UnpackOp unpackDefiningOp = dyn_cast_or_null<UnpackOp>(singleDefiningOp);

  if (!unpackDefiningOp)
    return failure();

  rewriter.replaceOp(op, unpackDefiningOp.getInput());
  return success();
}

LogicalResult SplitOp::verifyRegions() {
  SmallVector<Type> regTypes;
  if (failed(getRegTypes(*this, regTypes)))
    return failure();
  return verifyDefaultRegion(getOperation(), getRegion(), regTypes);
}

LogicalResult CombineOp::verifyRegions() {
  SmallVector<Type> regTypes;
  if (failed(getRegTypes(*this, regTypes)))
    return failure();
  return verifyDefaultRegion(getOperation(), getRegion(), regTypes);
}
