//===- StreamOps.td - Stream dialect ops -------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_STREAM_DIALECT_STREAM_OPS_TD
#define CIRCT_STREAM_DIALECT_STREAM_OPS_TD

include "circt-stream/Dialect/Stream/StreamDialect.td"
include "circt-stream/Dialect/Stream/StreamTypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypes.td"

// TODO: carefull with dependencies between elements
// TODO: add stronger verification constraints
def MapOp : Stream_Op<"map", [SingleBlockImplicitTerminator<"stream::YieldOp">]> {
    let summary = "Executes the region on each element";
    let description = [{
        TODO
    }];

    let arguments = (ins StreamType:$input);
    let results = (outs StreamType:$res);
    let regions = (region AnyRegion:$region);

    let assemblyFormat = [{
        `(` $input `)` attr-dict `:` functional-type($input, $res) $region
    }];
}

// TODO: add stronger verification constraints
def FilterOp : Stream_Op<"filter", [SingleBlockImplicitTerminator<"stream::YieldOp">]> {
    let summary = "Executes the region to check if an element should be preserved in the out stream";
    let description = [{
        TODO
    }];

    let arguments = (ins StreamType:$input);
    let results = (outs StreamType:$res);
    let regions = (region AnyRegion:$region);

    let assemblyFormat = [{
        `(` $input `)` attr-dict `:` functional-type($input, $res) $region
    }];
}

// TODO: add stronger verification constraints
def ReduceOp : Stream_Op<"reduce", [SingleBlockImplicitTerminator<"stream::YieldOp">]> {
    let summary = "applies the region on each stream element to reduce the stream to a single value";
    let description = [{
        TODO
    }];

    let arguments = (
      ins StreamType:$input,
      I64Attr:$initValue);

    let results = (outs StreamType:$res);
    let regions = (region AnyRegion:$region);

    let assemblyFormat = [{
        `(` $input `)` attr-dict `:` functional-type(operands, $res) $region
    }];
}

def UnpackOp : Stream_Op<"unpack", [
      NoSideEffect,
      TypesMatchWith<"result types match element types of 'tuple'",
                     "input", "results",
                     "$_self.cast<TupleType>().getTypes()">]> {

  let summary = "unpacks a tuple";
  let description = [{
    The `unpack` operation assignes each value of a tuple to a separate
    value for further processing. The number of results corresponds
    to the number of tuple elements.

    Example:
    ```mlir
    %a, %b = stream.unpack %tuple {attributes} : tuple<i32, i64>
    ```
  }];

  let arguments = (ins Builtin_Tuple:$input);

  let results = (outs Variadic<AnyType>:$results);

  let hasCustomAssemblyFormat = 1;
}

def PackOp : Stream_Op<"pack", [
  NoSideEffect,
  TypesMatchWith<"input types match element types of 'tuple'",
                   "result", "inputs",
                   "$_self.cast<TupleType>().getTypes()">
]> {
  let summary = "packs a tuple";
  let description = [{
    The `pack` operation constructs a tuple from separate values.
    The number of operands corresponds to the number of tuple elements.

    Example:
    ```mlir
    %tuple = stream.pack %a, %b {attributes} : tuple<i32, i64>
    ```
    }];

  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Builtin_Tuple:$result);

  let hasCustomAssemblyFormat = 1;
}

def YieldOp : Stream_Op<"yield", [NoSideEffect, ReturnLike, Terminator,
                               ParentOneOf<["MapOp, FilterOp, ReduceOp"]>]> {
  let summary = "stream yield and termination operation";
  let description = [{
    "stream.yield" yields an SSA value from the steam dialect op region and
    terminates the regions. The semantics of how the values are yielded is
    defined by the parent operation.
    If the parent operation defines no values, then the "scf.yield" may be
    left out in the custom syntax and the builders will insert one implicitly.
    Otherwise, it has to be present in the syntax to indicate which values are
    yielded.
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat =
      [{  attr-dict ($results^ `:` type($results))? }];
}

def CreateOp : Stream_Op<"create", [
  NoSideEffect
]> {
  let summary = "create a stream from an array attribute";
  let description = [{
    "stream.create" creates a stream from a provided array attribute.

    Currently, only integer streams are supported.

    Example:
    ```mlir
    %out = stream.create !stream.stream<i32> [1, 2, 3, 4]
    ```
    }];

  let arguments = (ins ArrayAttr:$values);
  let results = (outs StreamType:$result);

  let extraClassDeclaration = [{
    ::mlir::Type getElementType() {
      return this->result().getType().dyn_cast<StreamType>().getElementType();
    }
  }];


  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif // CIRCT_STREAM_DIALECT_STREAM_OPS_TD
