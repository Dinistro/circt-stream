//===-- Passes.td - Transforms pass definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes within the Transforms/ directory.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_STREAM_TRANSFORMS_PASSES
#define CIRCT_STREAM_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def CustomBufferInsertion : Pass<"custom-buffer-insertion", "circt::handshake::FuncOp"> {
  let summary = "Custom buffer insertion";
  let description = [{
    This pass inserts buffers everywhere, except in cycles. The goal of this pass is
    to increase the throughput, but reducing the latency of a cycle.
  }];
  let dependentDialects = ["circt::handshake::HandshakeDialect"];
  let options = [
    Option<"fifoBufferSize", "fifobuffer-size", "unsigned", /*default=*/"10",
           "Number of slots in each fifo buffer">,
  ];
}

#endif // CIRCT_STREAM_TRANSFORMS_PASSES

