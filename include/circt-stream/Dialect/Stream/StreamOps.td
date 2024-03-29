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

// TODO Refine this to ensure that each element holds data and a type
def RegisterAttr : OptionalAttr<ArrayAttr>;

def MapOp : Stream_Op<"map", [IsolatedFromAbove]> {
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

  let arguments = (ins StreamType:$input, RegisterAttr:$registers);
  let results = (outs StreamType:$res);
  let regions = (region AnyRegion:$region);

  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` functional-type($input, $res) $region
  }];

  let hasRegionVerifier = 1;
}

def FilterOp : Stream_Op<"filter", [IsolatedFromAbove]> {
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

  let arguments = (ins StreamType:$input, RegisterAttr:$registers);
  let results = (outs StreamType:$res);
  let regions = (region AnyRegion:$region);

  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` functional-type($input, $res) $region
  }];

  let hasRegionVerifier = 1;
}

def ReduceOp : Stream_Op<"reduce", [IsolatedFromAbove]> {
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
    I64Attr:$initValue,
    RegisterAttr:$registers);

  let results = (outs StreamType:$result);
  let regions = (region AnyRegion:$region);

  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` functional-type(operands, $result) $region
  }];

  let hasRegionVerifier = 1;
}

def UnpackOp : Stream_Op<"unpack", [
      Pure,
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

  let hasCanonicalizeMethod = 1;
}

def PackOp : Stream_Op<"pack", [
  Pure,
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

  let hasCanonicalizeMethod = 1;
}

def YieldOp : Stream_Op<"yield", [
    ReturnLike, Terminator,
    ParentOneOf<["MapOp, FilterOp, ReduceOp, SplitOp, CombineOp"]>
]> {
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

def SplitOp : Stream_Op<"split", [IsolatedFromAbove]> {
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

  let arguments = (ins StreamType:$input, RegisterAttr:$registers);
  let results = (outs Variadic<StreamType>:$results);
  let regions = (region AnyRegion:$region);

  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` functional-type($input, $results) $region
  }];

  let hasRegionVerifier = 1;
}

def CombineOp : Stream_Op<"combine", [IsolatedFromAbove]> {
  let summary = "combines elements from all incomming stream to one output element";
  let description = [{
    `stream.combine` takes a variable number of input stream and produces one output stream.
    When all input streams have an element ready, the provided region is executed to
    produce an output element.

    Example:
    ```mlir
    %res = stream.combine(%in0, %in1) : (!stream.stream<i32>, !stream.stream<i32>) -> (!stream.stream<tuple<i32, i32>>) {
    ^0(%val0: i32, %val1: i32):
      %0 = stream.pack %val0, %val1 : tuple<i32, i32>
      stream.yield %0 : tuple<i32, i32>
    }
    ```
    }];

  let arguments = (ins Variadic<StreamType>:$inputs, RegisterAttr:$registers);
  let results = (outs StreamType:$result);
  let regions = (region AnyRegion:$region);

  let assemblyFormat = [{
    `(` $inputs `)` attr-dict `:` functional-type($inputs, $result) $region
  }];

  let hasRegionVerifier = 1;
}

def SinkOp : Stream_Op<"sink", [
  Pure
]> {
  let summary = "consumes elements of the input stream";
  let description = [{
    "stream.sink" consumes all elements of the input stream while not having any output.

    Example:
    ```mlir
    stream.sink %in : !stream.stream<i64>
    ```
    }];

  let arguments = (ins StreamType:$input);

  let assemblyFormat = [{
    $input attr-dict `:` qualified(type($input))
  }];
}

#endif // CIRCT_STREAM_DIALECT_STREAM_OPS_TD
