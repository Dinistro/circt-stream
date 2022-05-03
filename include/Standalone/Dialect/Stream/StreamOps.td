//===- StandaloneOps.td - Standalone dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STREAM_OPS
#define STREAM_OPS

include "Standalone/Dialect/Stream/StreamDialect.td"
include "Standalone/Dialect/Stream/StreamTypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def StreamMin : Stream_Op<"min", [NoSideEffect,
        TypesMatchWith<"result type matches element type of 'stream'",
                     "input", "res",
                     "$_self.cast<StreamType>().getElementType()">]> {
    let summary = "Extracts the minimum of a finite stream.";
    let description = [{
        TODO
    }];

    let arguments = (ins StreamType:$input);
    let results = (outs AnyType:$res);

    let assemblyFormat = [{
        `(` $input `)` attr-dict `:` functional-type($input, $res)
    }];
}

def StreamMinContinuous : Stream_Op<"min_continuous", [NoSideEffect, SameOperandsAndResultType]> {
    let summary = "Extracts the minimum of a finite stream.";
    let description = [{
        The MinContinuous operation keeps track of the minimal value encountered
        so far by emit it continuously.
    }];

    let arguments = (ins StreamType:$input);
    let results = (outs StreamType:$res);

    let assemblyFormat = [{
        `(` $input `)` attr-dict `:` functional-type($input, $res)
    }];
}

// TODO: carefull with dependencies between elements
// TODO: add stronger verification constraints
def StreamMap : Stream_Op<"map", [SingleBlockImplicitTerminator<"stream::StreamYieldOp">]> {
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
def StreamFilter : Stream_Op<"filter", [SingleBlockImplicitTerminator<"stream::StreamYieldOp">]> {
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
def StreamReduce : Stream_Op<"reduce", [SingleBlockImplicitTerminator<"stream::StreamYieldOp">]> {
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

def StreamYieldOp : Stream_Op<"yield", [NoSideEffect, ReturnLike, Terminator,
                               ParentOneOf<["StreamMap, StreamFilter, StreamReduce"]>]> {
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


#endif // STREAM_OPS
