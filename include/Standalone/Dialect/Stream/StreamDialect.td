//===- StreamDialect.td - Stream dialect -------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STREAM_DIALECT
#define STREAM_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Stream dialect definition.
//===----------------------------------------------------------------------===//

def Stream_Dialect : Dialect {
    let name = "stream";
    let summary = "A dialect for stream processing.";
    let description = [{
        TODO
    }];
    let cppNamespace = "::mlir::stream";
}

//===----------------------------------------------------------------------===//
// Base stream operation definition.
//===----------------------------------------------------------------------===//

class Stream_Op<string mnemonic, list<Trait> traits = []> :
        Op<Stream_Dialect, mnemonic, traits>;

#endif // STREAM_DIALECT
