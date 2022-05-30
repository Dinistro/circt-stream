// REQUIRES: ieee-sim
// UNSUPPORTED: ieee-sim-iverilog
// RUN: standalone-opt %s --convert-stream-to-handshake \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=all --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --verilog > %reduce-export.sv
// RUN: circt-rtl-sim.py %reduce-export.sv %S/driverI64I1.sv --sim %ieee-sim --no-default-driver --top driver | FileCheck %s
// CHECK:      Element={{.*}}6
// CHECK-NEXT: EOS

module {
  func.func @top() -> !stream.stream<i64> {
    %in = stream.create !stream.stream<i64> [1,2,3]
    %out = stream.reduce(%in) {initValue = 0 : i64}: (!stream.stream<i64>) -> !stream.stream<i64> {
    ^0(%acc: i64, %val: i64):
      %r = arith.addi %acc, %val : i64
      stream.yield %r : i64
    }
    return %out : !stream.stream<i64>
  }
}
