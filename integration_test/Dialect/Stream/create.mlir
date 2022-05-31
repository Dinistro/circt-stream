// REQUIRES: ieee-sim
// UNSUPPORTED: ieee-sim-iverilog
// RUN: stream-opt %s --convert-stream-to-handshake \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=all --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --verilog > %create-export.sv
// RUN: circt-rtl-sim.py %create-export.sv %S/driverI64I1.sv --sim %ieee-sim --no-default-driver --top driver | FileCheck %s
// CHECK:      Element={{.*}}1
// CHECK-NEXT: Element={{.*}}2
// CHECK-NEXT: Element={{.*}}3
// CHECK-NEXT: EOS

module {
  func.func @top() -> !stream.stream<i64> {
    %out = stream.create !stream.stream<i64> [1,2,3]
    return %out : !stream.stream<i64>
  }
}
