// RUN: stream-opt %s --split-input-file --verify-diagnostics

func.func @wrong_reg_arg_type(%in: !stream.stream<i32>) -> !stream.stream<i32> {
  // expected-error @+1 {{expect the block argument #1 to have type 'i32', got 'i64' instead.}}
  %res = stream.map(%in) {registers = [0 : i32]}: (!stream.stream<i32>) -> !stream.stream<i32> {
  ^0(%val : i32, %reg: i64):
    %nReg = arith.addi %val, %val : i32
    stream.yield %nReg, %nReg : i32, i32
  }
  return %res : !stream.stream<i32>
}

// -----

func.func @wrong_reg_yield_type(%in: !stream.stream<i32>) -> !stream.stream<i32> {
  %res = stream.filter(%in) {registers = [0 : i32]}: (!stream.stream<i32>) -> !stream.stream<i32> {
  ^0(%val : i32, %reg: i32):
    %c0 = arith.constant 0 : i1
    // expected-error @+1 {{expect the operand #1 to have type 'i32', got 'i1' instead.}}
    stream.yield %c0, %c0 : i1, i1
  }
  return %res : !stream.stream<i32>
}

