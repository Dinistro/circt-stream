// RUN: stream-opt %s --split-input-file --verify-diagnostics

func.func @create_wrong_type() {
  // expected-error @+1 {{element #1's type does not match the type of the stream: expected 'i32' got 'i64'}}
  %0 = "stream.create"() {values = [1 : i32, 2 : i64, 3 : i32, 4 : i32]} : () -> !stream.stream<i32>
}

// -----

func.func @map_wrong_arg_types(%in: !stream.stream<tuple<i32, i32>>) -> (!stream.stream<i32>) {
  // expected-error @+1 {{expect the block argument #0 to have type 'tuple<i32, i32>', got 'i32' instead.}}
  %res = stream.map(%in) : (!stream.stream<tuple<i32, i32>>) -> !stream.stream<i32> {
  ^0(%val : i32):
    %0 = arith.constant 1 : i32
    %r = arith.addi %0, %val : i32
    stream.yield %r : i32
  }
  return %res : !stream.stream<i32>
}

// -----

func.func @map_wrong_arg_cnt(%in: !stream.stream<i32>) -> (!stream.stream<i32>) {
  // expected-error @+1 {{expect region to have 1 arguments.}}
  %res = stream.map(%in) : (!stream.stream<i32>) -> !stream.stream<i32> {
    ^0(%val : i32, %val2 : i64):
    stream.yield %val : i32
  }
  return %res : !stream.stream<i32>
}

// -----

func.func @filter_wrong_yield_type(%in: !stream.stream<i32>) -> (!stream.stream<i32>, !stream.stream<i32>) {
  %res = stream.filter(%in) : (!stream.stream<i32>) -> !stream.stream<i32> {
    ^0(%val : i32):
    // expected-error @+1 {{expect the operand #0 to have type 'i1', got 'i32' instead.}}
    stream.yield %val : i32
  }
  return %res : !stream.stream<i32>
}

// -----

func.func @reduce_wrong_arg_types(%in: !stream.stream<i64>) -> !stream.stream<i64> {
  // expected-error @+1 {{expect the block argument #0 to have type 'i64', got 'i32' instead.}}
  %res = stream.reduce(%in) {initValue = 0 : i64}: (!stream.stream<i64>) -> !stream.stream<i64> {
  ^0(%acc: i32, %val: i64):
    %0 = arith.constant 1 : i64
    stream.yield %0 : i64
  }
  return %res : !stream.stream<i64>
}

// -----

func.func @reduce_wrong_yield_type(%in: !stream.stream<i64>) -> !stream.stream<i64> {
  %res = stream.reduce(%in) {initValue = 0 : i64}: (!stream.stream<i64>) -> !stream.stream<i64> {
  ^0(%acc: i64, %val: i64):
    %0 = arith.constant 1 : i32
    // expected-error @+1 {{expect the operand #0 to have type 'i64', got 'i32' instead.}}
    stream.yield %0 : i32
  }
  return %res : !stream.stream<i64>
}

// -----

func.func @split_wrong_yield_args(%in: !stream.stream<tuple<i32, i32>>) -> (!stream.stream<i32>, !stream.stream<i32>) {
  %res0, %res1 = stream.split(%in) : (!stream.stream<tuple<i32, i32>>) -> (!stream.stream<i32>, !stream.stream<i32>) {
    ^0(%val0: tuple<i32, i32>):
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 0 : i64
    // expected-error @+1 {{expect the operand #1 to have type 'i32', got 'i64' instead.}}
    stream.yield %c0, %c1 : i32, i64
  }
  return %res0, %res1 : !stream.stream<i32>, !stream.stream<i32>
}

// -----

func.func @split_wrong_arg_types(%in: !stream.stream<tuple<i32, i32>>) -> (!stream.stream<i32>, !stream.stream<i32>) {
  // expected-error @+1 {{expect the block argument #0 to have type 'tuple<i32, i32>', got 'i32' instead.}}
  %res0, %res1 = stream.split(%in) : (!stream.stream<tuple<i32, i32>>) -> (!stream.stream<i32>, !stream.stream<i32>) {
  ^0(%val: i32):
    stream.yield %val, %val : i32, i32
  }
  return %res0, %res1 : !stream.stream<i32>, !stream.stream<i32>
}

// -----

func.func @split_wrong_arg_num(%in: !stream.stream<tuple<i32, i32>>) -> (!stream.stream<i32>, !stream.stream<i32>) {
  // expected-error @+1 {{expect region to have 1 arguments.}}
  %res0, %res1 = stream.split(%in) : (!stream.stream<tuple<i32, i32>>) -> (!stream.stream<i32>, !stream.stream<i32>) {
    ^0(%val0: i32, %val1: i32):
    stream.yield %val0, %val1 : i32, i32
  }
  return %res0, %res1 : !stream.stream<i32>, !stream.stream<i32>
}

// -----

func.func @split_wrong_yield_cnt(%in: !stream.stream<tuple<i32, i32>>) -> (!stream.stream<i32>, !stream.stream<i32>) {
  %res0, %res1 = stream.split(%in) : (!stream.stream<tuple<i32, i32>>) -> (!stream.stream<i32>, !stream.stream<i32>) {
    ^0(%val0: tuple<i32, i32>):
    %c = arith.constant 0 : i32
    // expected-error @+1 {{expect 2 operands, got 1}}
    stream.yield %c : i32
  }
  return %res0, %res1 : !stream.stream<i32>, !stream.stream<i32>
}

// -----

func.func @split_wrong_yield_args(%in: !stream.stream<tuple<i32, i32>>) -> (!stream.stream<i32>, !stream.stream<i32>) {
  %res0, %res1 = stream.split(%in) : (!stream.stream<tuple<i32, i32>>) -> (!stream.stream<i32>, !stream.stream<i32>) {
    ^0(%val0: tuple<i32, i32>):
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 0 : i64
    // expected-error @+1 {{expect the operand #1 to have type 'i32', got 'i64' instead.}}
    stream.yield %c0, %c1 : i32, i64
  }
  return %res0, %res1 : !stream.stream<i32>, !stream.stream<i32>
}
