// RUN: stream-opt %s --split-input-file --verify-diagnostics

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

func.func @filter_wrong_yield_type(%in: !stream.stream<i32>) -> !stream.stream<i32> {
  %res = stream.filter(%in) : (!stream.stream<i32>) -> !stream.stream<i32> {
    ^0(%val : i32):
    // expected-error @+1 {{expect the operand #0 to have type 'i1', got 'i32' instead.}}
    stream.yield %val : i32
  }
  return %res : !stream.stream<i32>
}

// -----

func.func @reduce_wrong_arg_types(%in: !stream.stream<i64>) -> !stream.stream<i8> {
  // expected-error @+1 {{expect the block argument #0 to have type 'i8', got 'i32' instead.}}
  %res = stream.reduce(%in) {initValue = 0 : i64}: (!stream.stream<i64>) -> !stream.stream<i8> {
  ^0(%acc: i32, %val: i64):
    %0 = arith.constant 1 : i8
    stream.yield %0 : i8
  }
  return %res : !stream.stream<i8>
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

// -----

func.func @combine_wrong_input_types(%in0: !stream.stream<i32>, %in1: !stream.stream<i32>) -> (!stream.stream<tuple<i32, i32>>) {
  // expected-error @+1 {{expect the block argument #1 to have type 'i32', got 'i64' instead.}}
    %res = stream.combine(%in0, %in1) : (!stream.stream<i32>, !stream.stream<i32>) -> (!stream.stream<tuple<i32, i32>>) {
    ^0(%val0: i32, %val1: i64):
      %0 = stream.pack %val0, %val0 : tuple<i32, i32>
      stream.yield %0 : tuple<i32, i32>
    }
    return %res : !stream.stream<tuple<i32, i32>>
  }

// -----

func.func @map_not_isolated(%in: !stream.stream<i32>) -> (!stream.stream<i32>) {
  %c = arith.constant 42 : i32
  // expected-note @+1 {{required by region isolation constrain}}
  %res = stream.map(%in) : (!stream.stream<i32>) -> !stream.stream<i32> {
  ^bb0(%val : i32):
    // expected-error @+1 {{op using value defined outside the region}}
    %out = arith.addi %val, %c : i32
    stream.yield %out : i32
  }
  return %res : !stream.stream<i32>
}
