// REQUIRES: verilator
// RUN: stream-opt %s --convert-stream-to-handshake \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=all --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --verilog > %t.sv && \
// RUN: circt-rtl-sim.py %t.sv %S/driver_out_i64.sv %S/driver.cpp --no-default-driver --top driver | FileCheck %s
// RUN: stream-opt %s --convert-stream-to-handshake \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=allFIFO --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --verilog > %t.sv && \
// RUN: circt-rtl-sim.py %t.sv %S/driver_out_i64.sv %S/driver.cpp --no-default-driver --top driver | FileCheck %s
// CHECK:      Element={{.*}}1
// CHECK-NEXT: Element={{.*}}0
// CHECK-NEXT: Element={{.*}}0
// CHECK-NEXT: EOS

module {
  func.func @top() -> !stream.stream<i64> {
    %in = stream.create !stream.stream<i64> [0,1,2,0,4,0]
    %out = stream.filter(%in) {registers = [0 : i1]}: (!stream.stream<i64>) -> !stream.stream<i64> {
    ^bb0(%val: i64, %reg: i1):
      %c1 = arith.constant 1 : i1
      %nReg = arith.xori %c1, %reg : i1
      stream.yield %reg, %nReg : i1, i1
    }
    return %out : !stream.stream<i64>
  }
}
