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

// CHECK:      Element=36
// CHECK-NEXT: EOS

module {
  func.func @top() -> !stream.stream<i64> {
    %in = stream.create !stream.stream<i64> [1,2,3]
    %out = stream.reduce(%in) {initValue = 0 : i64, registers = [10 : i64]}: (!stream.stream<i64>) -> !stream.stream<i64> {
    ^0(%acc: i64, %val: i64, %reg: i64):
      %r = arith.addi %acc, %val : i64
      %res = arith.addi %r, %reg : i64
      stream.yield %res, %reg : i64, i64
    }
    return %out : !stream.stream<i64>
  }
}
