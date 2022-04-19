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

#ifndef STANDALONE_CONVERSION_PASSES_TD
#define STANDALONE_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// StandaloneToScf
//===----------------------------------------------------------------------===//

def StandaloneToScf : Pass<"convert-standalone-to-scf", "mlir::func::FuncOp"> {
  let summary = "Convert the Standalone dialect to Scf";
  let constructor = "standalone::createStandaloneToScfPass()";
  let dependentDialects = [
    "standalone::StandaloneDialect",
    "scf::SCFDialect",
    "arith::ArithmeticDialect",
    "mlir::func::FuncDialect"
  ];
}

//===----------------------------------------------------------------------===//
// StreamToHandshake
//===----------------------------------------------------------------------===//

def StreamToHandshake : Pass<"convert-stream-to-handshake", "mlir::func::FuncOp"> {
  let summary = "Convert the Stream dialect to Handshake";
  let constructor = "stream::createStreamToHandshakePass()";
  let dependentDialects = [
    "stream::StreamDialect",
    "circt::handshake::HandshakeDialect",
    "arith::ArithmeticDialect",
    "mlir::func::FuncDialect"
  ];
}

#endif // STANDALONE_CONVERSION_PASSES_TD

