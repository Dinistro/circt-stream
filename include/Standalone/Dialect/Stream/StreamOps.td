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
include "mlir/Interfaces/SideEffectInterfaces.td"

def Stream_Min : Stream_Op<"min", [NoSideEffect,
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

#endif // STREAM_OPS
