// REQUIRES: cocotb, iverilog

// RUN: stream-opt %s --convert-stream-to-handshake \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=all --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --lowering-options=disallowLocalVariables --verilog > %t.sv && \
// RUN: %PYTHON% %S/../cocotb_driver.py --objdir=%t.sv.d/ --topLevel=top --pythonModule=projection-with-reg --pythonFolder=%S --testcase=all %t.sv 2>&1 | FileCheck %s

// CHECK: ** TEST
// CHECK: ** TESTS=[[CNT:.*]] PASS=[[CNT]] FAIL=0 SKIP=0
// CHECK: ********************************


module {
  func.func @top(%in: !stream.stream<tuple<i64, i64>>) -> !stream.stream<tuple<i64, i64>> {
    %out = stream.map(%in) {registers = [0 : i64] }: (!stream.stream<tuple<i64, i64>>) -> !stream.stream<tuple<i64, i64>> {
      ^0(%val : tuple<i64, i64>, %reg: i64):
      %l, %r = stream.unpack %val : tuple<i64, i64>
      %sum = arith.addi %l, %r : i64
      %c = arith.cmpi ult, %sum, %reg : i64
      %max = arith.select %c, %sum, %reg : i64
      %res = stream.pack %sum, %max : tuple<i64, i64>
      stream.yield %res, %max : tuple<i64, i64>, i64
    }
    return %out : !stream.stream<tuple<i64, i64>>
  }
}
