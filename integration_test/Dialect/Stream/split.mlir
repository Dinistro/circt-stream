// REQUIRES: cocotb, iverilog

// RUN: stream-opt %s --convert-stream-to-handshake \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=all --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --lowering-options=disallowLocalVariables --verilog > %t.sv && \
// RUN: %PYTHON% %S/cocotb_driver.py --objdir=%t.sv.d/ --topLevel=top --pythonModule=cocotb_bench --pythonFolder=%S --testcase=multipleOut %t.sv 2>&1 | FileCheck %s

// RUN: stream-opt %s --convert-stream-to-handshake \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=allFIFO --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --lowering-options=disallowLocalVariables --verilog > %t.sv && \
// RUN: %PYTHON% %S/cocotb_driver.py --objdir=%t.sv.d/ --topLevel=top --pythonModule=cocotb_bench --pythonFolder=%S --testcase=multipleOut %t.sv 2>&1 | FileCheck %s

// RUN: stream-opt %s --convert-stream-to-handshake \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=cycles --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --lowering-options=disallowLocalVariables --verilog > %t.sv && \
// RUN: %PYTHON% %S/cocotb_driver.py --objdir=%t.sv.d/ --topLevel=top --pythonModule=cocotb_bench --pythonFolder=%S --testcase=multipleOut %t.sv 2>&1 | FileCheck %s

// CHECK:      S0: Element=1
// CHECK-NEXT: S0: Element=2
// CHECK-NEXT: S0: Element=3
// CHECK-NEXT: S0: EOS
// CHECK-NEXT: S1: Element=11
// CHECK-NEXT: S1: Element=12
// CHECK-NEXT: S1: Element=13
// CHECK-NEXT: S1: EOS

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
