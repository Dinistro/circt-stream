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
static LogicalResult verifyRegion(Operation *op, Region &r) {
  SmallVector<Type> inputTypes =
      llvm::to_vector(llvm::map_range(op->getOperandTypes(), getElementType));

  SmallVector<Type> returnTypes =
      llvm::to_vector(llvm::map_range(op->getResultTypes(), getElementType));

  return verifyRegion(op, r, inputTypes, returnTypes);
}

LogicalResult MapOp::verifyRegions() {
  return verifyRegion(getOperation(), region());
}

LogicalResult FilterOp::verifyRegions() {
  SmallVector<Type> inputTypes = llvm::to_vector(
      llvm::map_range((*this)->getOperandTypes(), getElementType));

  Type boolType = IntegerType::get(this->getContext(), 1);

  return verifyRegion(getOperation(), region(), inputTypes, boolType);
}

LogicalResult ReduceOp::verifyRegions() {
  Type inputType = getElementType(input().getType());
  Type accType = getElementType(result().getType());

  return verifyRegion(getOperation(), region(), TypeRange({inputType, accType}),
                      accType);
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
  p << " " << input();
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << input().getType();
}

LogicalResult UnpackOp::canonicalize(UnpackOp op, PatternRewriter &rewriter) {

  // Replaces unnecessary `stream.unpack` when its operand is the result of a `stream.pack` operation.
  // In the following snippet, all uses of `%a2` and `%b2` are replaced with `%a` and `%b` respectively.
  //
  // ```
  //   %tuple = stream.pack %a, %b {attributes} : tuple<i32, i64>
  //   %a2, %b2 = stream.unpack %tuple {attributes} : tuple<i32, i64>
  //   // ... some ops using %a2, %b2
  // ```

  PackOp tuple = dyn_cast_or_null<PackOp>(op.input().getDefiningOp());
  if (!tuple)
    return failure();
  rewriter.replaceOp(op, tuple.inputs());
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
  p << " " << inputs();
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << result().getType();
}

LogicalResult PackOp::canonicalize(PackOp op, PatternRewriter &rewriter) {
  // Replaces an unnecessary `stream.pack` when it's operands are results of a `stream.unpack` op.
  // In the following snippet, all uses of `%tuple2` are replaced with `%tuple`.
  //
  // ```
  //   %a, %b = stream.unpack %tuple {attributes} : tuple<i32, i64>
  //   %tuple2 = stream.pack %a, %b {attributes} : tuple<i32, i64>
  //   // ... some ops using %tuple2
  // ```

  if (op.inputs().size() == 0)
    return failure();

  Operation *singleDefiningOp = op.inputs()[0].getDefiningOp();

  if (!llvm::all_of(op.inputs(), [singleDefiningOp](Value input) {
        return input.getDefiningOp() == singleDefiningOp;
      }))
    return failure();

  UnpackOp unpackDefiningOp = dyn_cast_or_null<UnpackOp>(singleDefiningOp);

  if (!unpackDefiningOp)
    return failure();
  
  rewriter.replaceOp(op, unpackDefiningOp.input());

  return success();
}

ParseResult CreateOp::parse(OpAsmParser &parser, OperationState &result) {
  if (failed(parser.parseOptionalAttrDict(result.attributes)))
    return failure();

  StreamType type;
  if (parser.parseType(type))
    return failure();

  result.addTypes(type);

  Type elementType = type.getElementType();
  if (!elementType.isIntOrIndex())
    return parser.emitError(parser.getNameLoc(),
                            "can only create streams of integers");

  SmallVector<Attribute> elements;
  if (parser.parseLSquare())
    return failure();

  if (parser.parseCommaSeparatedList([&]() {
        APInt element(elementType.getIntOrFloatBitWidth(), 0);
        if (parser.parseInteger(element))
          return failure();
        elements.push_back(IntegerAttr::get(elementType, element));
        return success();
      }))
    return failure();

  if (parser.parseRSquare())
    return failure();

  result.addAttribute("values", ArrayAttr::get(parser.getContext(), elements));
  return success();
}

void CreateOp::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs(), {"values"});
  p << " ";
  p << result().getType();
  p << " [";
  llvm::interleaveComma(values(), p, [&](Attribute attr) {
    assert(attr.isa<IntegerAttr>() && "can only handle integer attributes");

    auto intAttr = attr.dyn_cast<IntegerAttr>();

    p << intAttr.getValue();
  });
  p << "]";
}

LogicalResult CreateOp::verify() {
  StreamType type = result().getType().dyn_cast<StreamType>();
  assert(type && "the type constraint should ensure that we get a StreamType");

  Type elementType = type.getElementType();

  for (auto it : llvm::enumerate(values())) {
    unsigned i = it.index();
    auto attr = it.value();
    auto intAttr = attr.dyn_cast<IntegerAttr>();
    if (!intAttr)
      return emitError("element #") << i << " is not an integer attribute";

    if (intAttr.getType() != elementType)
      return emitError("element #")
             << i << "'s type does not match the type of the stream: expected "
             << elementType << " got " << intAttr.getType();
  }

  // TODO ensure that all array elements have the same type
  return success();
}

LogicalResult SplitOp::verifyRegions() {
  return verifyRegion(getOperation(), region());
}

LogicalResult CombineOp::verifyRegions() {
  return verifyRegion(getOperation(), region());
}
