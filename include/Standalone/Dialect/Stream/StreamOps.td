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

def Stream_FooOp : Stream_Op<"foo", [NoSideEffect,
                                             SameOperandsAndResultType]> {
    let summary = "TODO";
    let description = [{
        TODO
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

#endif // STREAM_OPS
