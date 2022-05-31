//===- StreamDialect.td - Stream dialect -------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_STREAM_DIALECT_STREAM_TD
#define CIRCT_STREAM_DIALECT_STREAM_TD

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

    let useDefaultTypePrinterParser = 1;
    //let useDefaultAttributePrinterParser = 1;

    let cppNamespace = "::circt_stream::stream";
}

//===----------------------------------------------------------------------===//
// Base stream operation definition.
//===----------------------------------------------------------------------===//

class Stream_Op<string mnemonic, list<Trait> traits = []> :
        Op<Stream_Dialect, mnemonic, traits>;

#endif // CIRCT_STREAM_DIALECT_STREAM_TD
