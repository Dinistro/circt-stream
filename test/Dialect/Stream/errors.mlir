// RUN: stream-opt %s --split-input-file --verify-diagnostics

func.func @create_wrong_type() {
  // expected-error @+1 {{element #1's type does not match the type of the stream: expected 'i32' got 'i64'}}
  %0 = "stream.create"() {values = [1 : i32, 2 : i64, 3 : i32, 4 : i32]} : () -> !stream.stream<i32>
}

// -----

func.func @split_wrong_arg_types(%in: !stream.stream<tuple<i32, i32>>) -> (!stream.stream<i32>, !stream.stream<i32>) {
  // expected-error @+1 {{expect the block argument to have type 'tuple<i32, i32>', got 'i32' instead.}}
  %res0, %res1 = stream.split(%in) : (!stream.stream<tuple<i32, i32>>) -> (!stream.stream<i32>, !stream.stream<i32>) {
  ^0(%val: i32):
    stream.yield %val, %val : i32, i32
  }
  return %res0, %res1 : !stream.stream<i32>, !stream.stream<i32>
}

// -----

func.func @split_wrong_arg_num(%in: !stream.stream<tuple<i32, i32>>) -> (!stream.stream<i32>, !stream.stream<i32>) {
  // expected-error @+1 {{expect region to have exactly one argument.}}
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
    // expected-error @+1 {{expect the return types to match the types of the output streams}}
    stream.yield %c0, %c1 : i32, i64
  }
  return %res0, %res1 : !stream.stream<i32>, !stream.stream<i32>
}
