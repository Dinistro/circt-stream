//===- StreamTypes.td - Stream Types -----------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STREAM_TYPES
#define STREAM_TYPES

include "mlir/IR/AttrTypeBase.td"
include "Standalone/Dialect/Stream/StreamDialect.td"

class StreamDialect_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Stream_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def StreamType : StreamDialect_Type<"Stream", "stream"> {
  let summary = "A type for streams with elements of type elementType";

  let parameters = (ins "::mlir::Type":$elementType);

  let assemblyFormat = "`<` $elementType `>`";
}

#endif // STREAM_TYPES
