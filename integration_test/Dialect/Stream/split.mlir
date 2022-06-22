// REQUIRES: verilator
// RUN: stream-opt %s --convert-stream-to-handshake \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=all --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --verilog > %t.sv && \
// RUN: circt-rtl-sim.py %t.sv %S/driver_out_i64_i64.sv %S/driver.cpp --no-default-driver --top driver | FileCheck %s
// RUN: stream-opt %s --convert-stream-to-handshake \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=allFIFO --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --verilog > %t.sv && \
// RUN: circt-rtl-sim.py %t.sv %S/driver_out_i64_i64.sv %S/driver.cpp --no-default-driver --top driver | FileCheck %s
// CHECK-DAG: S0: Element={{.*}}1
// CHECK-DAG: S0: Element={{.*}}2
// CHECK-DAG: S0: Element={{.*}}3
// CHECK-DAG: S1: Element={{.*}}11
// CHECK-DAG: S1: Element={{.*}}12
// CHECK-DAG: S1: Element={{.*}}13
// CHECK-NOT: ensures that the later appears after the former
// CHECK-DAG: S0: EOS
// CHECK-DAG: S1: EOS

module {
  func.func @top() -> (!stream.stream<i64>, !stream.stream<i64>) {
    %in = stream.create !stream.stream<i64> [1,2,3]
    %res0, %res1 = stream.split(%in) : (!stream.stream<i64>) -> (!stream.stream<i64>, !stream.stream<i64>) {
    ^0(%val: i64):
      %c10 = arith.constant 10 : i64
      %snd = arith.addi %val, %c10 : i64
      stream.yield %val, %snd : i64, i64
    }
    return %res0, %res1 : !stream.stream<i64>, !stream.stream<i64>
  }
}
