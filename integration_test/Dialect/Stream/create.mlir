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

// CHECK: Element=1
// CHECK: Element=2
// CHECK: Element=3
// CHECK: EOS

module {
  func.func @top() -> !stream.stream<i64> {
    %out = stream.create !stream.stream<i64> [1,2,3]
    return %out : !stream.stream<i64>
  }
}
