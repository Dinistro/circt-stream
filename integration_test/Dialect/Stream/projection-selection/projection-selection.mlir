// REQUIRES: cocotb, iverilog

// RUN: stream-opt %s --convert-stream-to-handshake \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=all --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --lowering-options=disallowLocalVariables --verilog > %t.sv && \
// RUN: %PYTHON% %S/../cocotb_driver.py --objdir=%t.sv.d/ --topLevel=top --pythonModule=projection-selection --pythonFolder=%S %t.sv 2>&1 | FileCheck %s

// CHECK: ** TEST
// CHECK: ** TESTS=[[N:.*]] PASS=[[N]] FAIL=0 SKIP=0


!T = tuple<i64, i64, i64, i64, i64, i64, i64, i64>
!Tout = tuple<i64, i64>

module {
  func.func @top(%in: !stream.stream<!T>) -> !stream.stream<!Tout> {
   %mapOut = stream.map(%in) : (!stream.stream<!T>) -> !stream.stream<!Tout> {
    ^0(%val : !T):
      %e:8 = stream.unpack %val : !T
      %f0 = arith.addi %e#0, %e#1 : i64
      %f1 = arith.addi %e#4, %e#5 : i64
      %res = stream.pack %f0, %f1 : !Tout
      stream.yield %res : !Tout
    }
    %out = stream.filter(%mapOut) : (!stream.stream<!Tout>) -> !stream.stream<!Tout> {
    ^0(%val : !Tout):
      %e:2 = stream.unpack %val : !Tout
      %cond = arith.cmpi sle, %e#0, %e#1 : i64
      stream.yield %cond : i1
    }
    return %out : !stream.stream<!Tout>
  }
}
