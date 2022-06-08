// REQUIRES: ieee-sim
// RUN: stream-opt %s --convert-stream-to-handshake \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=all --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --verilog > %map-export.sv && \
// RUN: circt-rtl-sim.py %map-export.sv %S/driver_out_i64.sv --sim %ieee-sim --no-default-driver --top driver | FileCheck %s
// CHECK:      Element={{.*}}11
// CHECK-NEXT: Element={{.*}}12
// CHECK-NEXT: Element={{.*}}13
// CHECK-NEXT: EOS

module {
  func.func @top() -> !stream.stream<i64> {
    %in = stream.create !stream.stream<i64> [1,2,3]
    %out = stream.map(%in) : (!stream.stream<i64>) -> !stream.stream<i64> {
    ^0(%val : i64):
      %0 = arith.constant 10 : i64
      %r = arith.addi %0, %val : i64
      stream.yield %r : i64
    }
    return %out : !stream.stream<i64>
  }
}
