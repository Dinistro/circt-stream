//===-- Passes.td - Conversion pass definitions ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for all dialect conversions.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_STREAM_CONVERSION_PASSES_TD
#define CIRCT_STREAM_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// StreamToHandshake
//===----------------------------------------------------------------------===//

def StreamToHandshake : Pass<"convert-stream-to-handshake", "mlir::ModuleOp"> {
  let summary = "Convert the Stream dialect to Handshake";
  let constructor = "circt_stream::createStreamToHandshakePass()";
  let dependentDialects = [
    "circt_stream::stream::StreamDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::func::FuncDialect",
    "circt::handshake::HandshakeDialect"
  ];
}

#endif // CIRCT_STREAM_CONVERSION_PASSES_TD

