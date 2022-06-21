// REQUIRES: verilator
// RUN: stream-opt %s --convert-stream-to-handshake \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=all --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --verilog > %t.sv && \
// RUN: circt-rtl-sim.py %t.sv %S/driver_out_i64.sv %S/driver.cpp --no-default-driver --top driver | FileCheck %s
// CHECK:      Element={{.*}}1
// CHECK-NEXT: Element={{.*}}2
// CHECK-NEXT: Element={{.*}}4
// CHECK-NEXT: EOS

module {
  func.func @top() -> !stream.stream<i64> {
    %in = stream.create !stream.stream<i64> [0,1,2,0,4,0]
    %out = stream.filter(%in) : (!stream.stream<i64>) -> !stream.stream<i64> {
    ^bb0(%val: i64):
      %c0_i64 = arith.constant 0 : i64
      %0 = arith.cmpi sgt, %val, %c0_i64 : i64
      stream.yield %0 : i1
    }
    return %out : !stream.stream<i64>
  }
}
