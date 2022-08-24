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

// CHECK:      Element=100
// CHECK:      Element=1
// CHECK:      Element=100
// CHECK-NEXT: EOS

module {
  func.func @top() -> !stream.stream<i64> {
    %in = stream.create !stream.stream<i64> [0,1,0]
    %out = stream.map(%in) : (!stream.stream<i64>) -> (!stream.stream<i64>) {
    ^0(%val: i64):
      %c0 = arith.constant 0 : i64
      %cond = arith.cmpi eq, %c0, %val : i64
      cf.cond_br %cond, ^1(%c0: i64), ^2(%val: i64)
    ^1(%i: i64):
      %c10 = arith.constant 100 : i64
      %lcond = arith.cmpi eq, %c10, %i : i64
      cf.cond_br %lcond, ^2(%i: i64), ^body
    ^body:
      %c1 = arith.constant 1 : i64
      %ni = arith.addi %i, %c1 : i64
      cf.br ^1(%ni: i64)
    ^2(%res: i64):
      stream.yield %res : i64
    }

    return %out : !stream.stream<i64>
  }
}
