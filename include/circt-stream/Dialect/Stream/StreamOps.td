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

// TODO: add stronger verification constraints
def MapOp : Stream_Op<"map", [SingleBlockImplicitTerminator<"stream::YieldOp">]> {
  let summary = "applies the region on each element";
  let description = [{
    `stream.map` applies the provided region on each element of the input
    stream.
    The result will be emitted on the output stream.

    Example:
    ```mlir
    %res = stream.map(%in) : (!stream.stream<i32>) -> !stream.stream<i32> {
    ^0(%val : i32):
      %0 = arith.constant 1 : i32
      %r = arith.addi %0, %val : i32
      stream.yield %r : i32
    }
    ```
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
  let summary = "filters a stream with the provided predicate";
  let description = [{
    `stream.filter` applies the provided region on each element of the input
    stream.
    If the result is true/1, then the input element is forwarded to the output,
    otherwise it's dropped.

    Example:
    ```mlir
    %out = stream.filter(%in) : (!stream.stream<i32>) -> !stream.stream<i32> {
    ^bb0(%val: i32):
      %c0_i32 = arith.constant 0 : i32
      %0 = arith.cmpi sgt, %val, %c0_i32 : i32
      stream.yield %0 : i1
    }
    ```
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
  let summary = "reduces the stream with the provided region";
  let description = [{
    `stream.reduce` folds the stream to a single value by applying the provided
    region on each element. The result of one such application is provided to
    the next one as the first parameter.

    Example:
    ```mlir
    %res = stream.reduce(%in) {initValue = 0 : i64}: (!stream.stream<i64>) -> !stream.stream<i64> {
    ^0(%acc: i64, %val: i64):
      %r = arith.addi %acc, %val : i64
      stream.yield %r : i64
    }
    ```
  }];

  let arguments = (
    ins StreamType:$input,
    I64Attr:$initValue);

  let results = (outs StreamType:$result);
  let regions = (region AnyRegion:$region);

  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` functional-type(operands, $result) $region
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
                               ParentOneOf<["MapOp, FilterOp, ReduceOp, SplitOp"]>]> {
  let summary = "stream yield and termination operation";
  let description = [{
    "stream.yield" yields an SSA value from the steam dialect op region and
    terminates the regions. The semantics of how the values are yielded is
    defined by the parent operation.
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

    **NOTE**: Currently, only integer streams are supported.

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

def SplitOp : Stream_Op<"split", [SingleBlockImplicitTerminator<"stream::YieldOp">]> {
  let summary = "for each input produces outputs for each output stream.";
  let description = [{
    `stream.split` takes one input stream and produces multiple output streams.
    For each input element, each output stream expects an element. The exact
    splitting is implemented in a region.

    Example:
    ```mlir
    %res0, %res1 = stream.split(%in) : (!stream.stream<tuple<i32, i32>>) -> (!stream.stream<i32>, !stream.stream<i32>) {
    ^0(%val: tuple<i32, i32>):
      %0, %1 = stream.unpack %val : tuple<i32, i32>
      stream.yield %0, %1 : i32, i32
    }
    ```
    }];

  let arguments = (ins StreamType:$input);
  let results = (outs Variadic<StreamType>:$results);
  let regions = (region AnyRegion:$region);

  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` functional-type($input, $results) $region
  }];

  let hasRegionVerifier = 1;
}

#endif // CIRCT_STREAM_DIALECT_STREAM_OPS_TD
