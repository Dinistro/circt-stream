// REQUIRES: cocotb, iverilog

// RUN: stream-opt %s --convert-stream-to-handshake \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=all --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --lowering-options=disallowLocalVariables --verilog > %t.sv && \
// RUN: %PYTHON% %S/../cocotb_driver.py --objdir=%t.sv.d/ --topLevel=top --pythonModule=projection --pythonFolder=%S --testcase=all %t.sv 2>&1 | FileCheck %s

// CHECK:      ** TEST
// CHECK-NEXT: ********************************
// CHECK-NEXT: ** projection.increase
// CHECK-NEXT: ********************************
// CHECK-NEXT: ** TESTS=1 PASS=1 FAIL=0 SKIP=0
// CHECK-NEXT: ********************************


module {
  func.func @top(%in: !stream.stream<tuple<i64, i64>>) -> !stream.stream<i64> {
    %out = stream.map(%in) : (!stream.stream<tuple<i64, i64>>) -> !stream.stream<i64> {
    ^0(%val : tuple<i64, i64>):
      %l, %r = stream.unpack %val : tuple<i64, i64>
      %res = arith.addi %l, %r : i64
      stream.yield %res : i64
    }
    return %out : !stream.stream<i64>
  }
}
