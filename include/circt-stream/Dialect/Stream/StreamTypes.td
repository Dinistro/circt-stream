//===- StreamTypes.td - Stream Types -----------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_STREAM_DIALECT_STREAM_TYPES_TD
#define CIRCT_STREAM_DIALECT_STREAM_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "circt-stream/Dialect/Stream/StreamDialect.td"

class StreamDialect_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Stream_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def StreamType : StreamDialect_Type<"Stream", "stream"> {
  let summary = "A type for streams with elements of type elementType";
  let description = [{
    Parameterized stream type that is used to model streams with a fixed
    element type.
  }];

  let parameters = (ins "::mlir::Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}

#endif // CIRCT_STREAM_DIALECT_STREAM_TYPES_TD
