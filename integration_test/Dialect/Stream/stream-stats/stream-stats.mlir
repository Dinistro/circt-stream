// REQUIRES: cocotb, iverilog

// RUN: stream-opt %s --convert-stream-to-handshake \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=all --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --lowering-options=disallowLocalVariables --verilog > %t.sv && \
// RUN: %PYTHON% %S/../cocotb_driver.py --objdir=%t.sv.d/ --topLevel=top --pythonModule=stream-stats --pythonFolder=%S %t.sv 2>&1 | FileCheck %s

// RUN: stream-opt %s --convert-stream-to-handshake \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=allFIFO --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --lowering-options=disallowLocalVariables --verilog > %t.sv && \
// RUN: %PYTHON% %S/../cocotb_driver.py --objdir=%t.sv.d/ --topLevel=top --pythonModule=stream-stats --pythonFolder=%S %t.sv 2>&1 | FileCheck %s

// RUN: stream-opt %s --convert-stream-to-handshake \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=cycles --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --lowering-options=disallowLocalVariables --verilog > %t.sv && \
// RUN: %PYTHON% %S/../cocotb_driver.py --objdir=%t.sv.d/ --topLevel=top --pythonModule=stream-stats --pythonFolder=%S %t.sv 2>&1 | FileCheck %s

// RUN: stream-opt %s --convert-stream-to-handshake \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --custom-buffer-insertion --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --lowering-options=disallowLocalVariables --verilog > %t.sv && \
// RUN: %PYTHON% %S/../cocotb_driver.py --objdir=%t.sv.d/ --topLevel=top --pythonModule=stream-stats --pythonFolder=%S %t.sv 2>&1 | FileCheck %s

// CHECK: ** TEST
// CHECK: ** TESTS=[[N:.*]] PASS=[[N]] FAIL=0 SKIP=0

!T = tuple<i64, i64, i64, i64, i64, i64, i64, i64>
module {
  func.func @top(%in: !stream.stream<!T>) -> (!stream.stream<!T>, !stream.stream<i64>) {
    %data, %copy = stream.split(%in) : (!stream.stream<!T>) -> (!stream.stream<!T>, !stream.stream<!T>) {
    ^0(%val : !T):
      stream.yield %val, %val : !T, !T
    }

    %maxE = stream.map(%copy) : (!stream.stream<!T>) -> (!stream.stream<i64>) {
    ^0(%val : !T):
      %e:8 = stream.unpack %val : !T

      %c0 = arith.cmpi slt, %e#0, %e#1 : i64
      %t0 = arith.select %c0, %e#0, %e#1 : i64
      %c1 = arith.cmpi slt, %e#2, %e#3 : i64
      %t1 = arith.select %c1, %e#2, %e#3 : i64
      %c2 = arith.cmpi slt, %e#4, %e#5 : i64
      %t2 = arith.select %c2, %e#4, %e#5 : i64
      %c3 = arith.cmpi slt, %e#6, %e#7 : i64
      %t3 = arith.select %c3, %e#6, %e#7 : i64

      %c4 = arith.cmpi slt, %t0, %t1 : i64
      %t4 = arith.select %c4, %t0, %t1 : i64
      %c5 = arith.cmpi slt, %t2, %t3 : i64
      %t5 = arith.select %c5, %t2, %t3 : i64

      %c6 = arith.cmpi slt, %t4, %t5 : i64
      %t6 = arith.select %c6, %t4, %t5 : i64

      stream.yield %t6 : i64
    }

    %max = stream.reduce(%maxE) {initValue = 0 : i64}: (!stream.stream<i64>) -> !stream.stream<i64> {
    ^0(%acc: i64, %val: i64):
      %pred = arith.cmpi slt, %acc, %val : i64
      %newAcc = arith.select %pred, %acc, %val : i64
      stream.yield %newAcc : i64
    }

    return %data, %max: !stream.stream<!T>, !stream.stream<i64>
  }
}
