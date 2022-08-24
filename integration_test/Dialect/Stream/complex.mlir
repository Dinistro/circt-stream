// REQUIRES: cocotb, iverilog

// RUN: stream-opt %s --convert-stream-to-handshake \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=all --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --lowering-options=disallowLocalVariables --verilog > %t.sv && \
// RUN: %PYTHON% %S/cocotb_driver.py --objdir=%t.sv.d/ --topLevel=top --pythonModule=cocotb_bench --pythonFolder=%S %t.sv 2>&1 | FileCheck %s

// RUN: stream-opt %s --convert-stream-to-handshake \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=allFIFO --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --lowering-options=disallowLocalVariables --verilog > %t.sv && \
// RUN: %PYTHON% %S/cocotb_driver.py --objdir=%t.sv.d/ --topLevel=top --pythonModule=cocotb_bench --pythonFolder=%S %t.sv 2>&1 | FileCheck %s

// RUN: stream-opt %s --convert-stream-to-handshake \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=cycles --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --lowering-options=disallowLocalVariables --verilog > %t.sv && \
// RUN: %PYTHON% %S/cocotb_driver.py --objdir=%t.sv.d/ --topLevel=top --pythonModule=cocotb_bench --pythonFolder=%S %t.sv 2>&1 | FileCheck %s

// CHECK:      Element=12
// CHECK-NEXT: EOS

module {
  func.func @top() -> !stream.stream<i64> {
    %in = stream.create !stream.stream<i64> [1,2,3]
    %left, %right = stream.split(%in) : (!stream.stream<i64>) -> (!stream.stream<i64>, !stream.stream<i64>) {
    ^0(%val : i64):
      stream.yield %val, %val : i64, i64
    }

    %leftR = stream.reduce(%left) {initValue = 0 : i64}: (!stream.stream<i64>) -> !stream.stream<i64> {
    ^0(%acc: i64, %val: i64):
      %r = arith.addi %acc, %val : i64
      stream.yield %r : i64
    }

    %rightR = stream.reduce(%right) {initValue = 1 : i64}: (!stream.stream<i64>) -> !stream.stream<i64> {
    ^0(%acc: i64, %val: i64):
      %r = arith.muli %acc, %val : i64
      stream.yield %r : i64
    }

    %out = stream.combine(%leftR, %rightR) : (!stream.stream<i64>, !stream.stream<i64>) -> (!stream.stream<i64>) {
    ^0(%val0: i64, %val1: i64):
      %0 = arith.addi %val0, %val1 : i64
      stream.yield %0 : i64
    }

    return %out : !stream.stream<i64>
  }
}
