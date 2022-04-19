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

#endif // STREAM_OPS
