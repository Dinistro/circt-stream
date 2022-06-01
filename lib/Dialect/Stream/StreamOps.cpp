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

ParseResult UnpackOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand tuple;
  TupleType type;

  if (parser.parseOperand(tuple) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(type))
    return failure();

  if (parser.resolveOperand(tuple, type, result.operands)) return failure();

  result.addTypes(type.getTypes());

  return success();
}

void UnpackOp::print(OpAsmPrinter &p) {
  p << " " << input();
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << input().getType();
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

ParseResult CreateOp::parse(OpAsmParser &parser, OperationState &result) {
  if (failed(parser.parseOptionalAttrDict(result.attributes))) return failure();

  StreamType type;
  if (parser.parseType(type)) return failure();

  result.addTypes(type);

  Type elementType = type.getElementType();
  if (!elementType.isIntOrIndex())
    return parser.emitError(parser.getNameLoc(),
                            "can only create streams of integers");

  SmallVector<Attribute> elements;
  if (parser.parseLSquare()) return failure();

  if (parser.parseCommaSeparatedList([&]() {
        APInt element(elementType.getIntOrFloatBitWidth(), 0);
        if (parser.parseInteger(element)) return failure();
        elements.push_back(IntegerAttr::get(elementType, element));
        return success();
      }))
    return failure();

  if (parser.parseRSquare()) return failure();

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
