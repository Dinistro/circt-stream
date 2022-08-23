// REQUIRES: verilator
// RUN: stream-opt %s --convert-stream-to-handshake \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=all --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --verilog > %t.sv && \
// RUN: circt-rtl-sim.py %t.sv %S/driver_out_i64.sv %S/driver.cpp --no-default-driver --top driver | FileCheck %s

// CHECK:      Element={{.*}}1
// CHECK-NEXT: Element={{.*}}3
// CHECK-NEXT: Element={{.*}}6
// CHECK-NEXT: EOS

module {
  func.func @top() -> !stream.stream<i64> {
    %in = stream.create !stream.stream<i64> [1,2,3]
    %res = stream.map(%in) {registers = [0: i64]}: (!stream.stream<i64>) -> !stream.stream<i64> {
      ^0(%val : i64, %reg: i64):
      %nReg = arith.addi %val, %reg : i64
      stream.yield %nReg, %nReg : i64, i64
    }
    return %res : !stream.stream<i64>
  }
}
