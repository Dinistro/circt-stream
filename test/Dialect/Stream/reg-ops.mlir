// RUN: stream-opt %s --mlir-print-op-generic | stream-opt | FileCheck %s

// CHECK-LABEL:   func.func @map(
// CHECK-SAME:                   %[[VAL_0:.*]]: !stream.stream<i32>) -> !stream.stream<i32> {
// CHECK:           %[[VAL_1:.*]] = stream.map(%[[VAL_0]]) {registers = [0 : i32]} : (!stream.stream<i32>) -> !stream.stream<i32> {
// CHECK:           ^bb0(%[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32):
// CHECK:             %[[VAL_4:.*]] = arith.addi %[[VAL_2]], %[[VAL_3]] : i32
// CHECK:             stream.yield %[[VAL_4]], %[[VAL_4]] : i32, i32
// CHECK:           }
// CHECK:           return %[[VAL_5:.*]] : !stream.stream<i32>
// CHECK:         }

func.func @map(%in: !stream.stream<i32>) -> !stream.stream<i32> {
  %res = stream.map(%in) {registers = [0 : i32]}: (!stream.stream<i32>) -> !stream.stream<i32> {
  ^0(%val : i32, %reg: i32):
    %nReg = arith.addi %val, %reg : i32
    stream.yield %nReg, %nReg : i32, i32
  }
  return %res : !stream.stream<i32>
}

// CHECK-LABEL:   func.func @filter(
// CHECK-SAME:                      %[[VAL_0:.*]]: !stream.stream<i32>) -> !stream.stream<i32> {
// CHECK:           %[[VAL_1:.*]] = stream.filter(%[[VAL_0]]) {registers = [false]} : (!stream.stream<i32>) -> !stream.stream<i32> {
// CHECK:           ^bb0(%[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i1):
// CHECK:             %[[VAL_4:.*]] = arith.constant true
// CHECK:             %[[VAL_5:.*]] = arith.xori %[[VAL_4]], %[[VAL_3]] : i1
// CHECK:             stream.yield %[[VAL_5]], %[[VAL_5]] : i1, i1
// CHECK:           }
// CHECK:           return %[[VAL_6:.*]] : !stream.stream<i32>
// CHECK:         }

func.func @filter(%in: !stream.stream<i32>) -> !stream.stream<i32> {
  %res = stream.filter(%in) {registers = [0 : i1]}: (!stream.stream<i32>) -> !stream.stream<i32> {
  ^0(%val : i32, %reg: i1):
    %c1 = arith.constant 1 : i1
    %nReg = arith.xori %c1, %reg : i1
    stream.yield %nReg, %nReg : i1, i1
  }
  return %res : !stream.stream<i32>
}

// CHECK-LABEL:   func.func @reduce(
// CHECK-SAME:                      %[[VAL_0:.*]]: !stream.stream<i64>) -> !stream.stream<i64> {
// CHECK:           %[[VAL_1:.*]] = stream.reduce(%[[VAL_0]]) {initValue = 0 : i64, registers = [0]} : (!stream.stream<i64>) -> !stream.stream<i64> {
// CHECK:           ^bb0(%[[VAL_2:.*]]: i64, %[[VAL_3:.*]]: i64, %[[VAL_4:.*]]: i64):
// CHECK:             %[[VAL_5:.*]] = arith.addi %[[VAL_2]], %[[VAL_3]] : i64
// CHECK:             stream.yield %[[VAL_5]], %[[VAL_4]] : i64, i64
// CHECK:           }
// CHECK:           return %[[VAL_6:.*]] : !stream.stream<i64>
// CHECK:         }

func.func @reduce(%in: !stream.stream<i64>) -> !stream.stream<i64> {
  %res = stream.reduce(%in) {initValue = 0 : i64, registers = [0 : i64]}: (!stream.stream<i64>) -> !stream.stream<i64> {
  ^0(%acc: i64, %val: i64, %reg: i64):
    %r = arith.addi %acc, %val : i64
    stream.yield %r, %reg : i64, i64
  }
  return %res : !stream.stream<i64>
}

// CHECK-LABEL:   func.func @split(
// CHECK-SAME:                     %[[VAL_0:.*]]: !stream.stream<i32>) -> (!stream.stream<i32>, !stream.stream<i32>) {
// CHECK:           %[[VAL_1:.*]]:2 = stream.split(%[[VAL_0]]) {registers = [0, true, 42 : i32]} : (!stream.stream<i32>) -> (!stream.stream<i32>, !stream.stream<i32>) {
// CHECK:           ^bb0(%[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i64, %[[VAL_4:.*]]: i1, %[[VAL_5:.*]]: i32):
// CHECK:             stream.yield %[[VAL_2]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_5]] : i32, i32, i64, i1, i32
// CHECK:           }
// CHECK:           return %[[VAL_6:.*]]#0, %[[VAL_6]]#1 : !stream.stream<i32>, !stream.stream<i32>
// CHECK:         }

func.func @split(%in: !stream.stream<i32>) -> (!stream.stream<i32>, !stream.stream<i32>) {
  %res0, %res1 = stream.split(%in) {registers = [0 : i64, 1 : i1, 42 : i32]} : (!stream.stream<i32>) -> (!stream.stream<i32>, !stream.stream<i32>) {
    ^0(%val: i32, %reg0: i64, %reg1 : i1, %reg2: i32):
    stream.yield %val, %val, %reg0, %reg1, %reg2 : i32, i32, i64, i1, i32
  }
  return %res0, %res1 : !stream.stream<i32>, !stream.stream<i32>
}

// CHECK-LABEL:   func.func @combine(
// CHECK-SAME:                       %[[VAL_0:.*]]: !stream.stream<i32>,
// CHECK-SAME:                       %[[VAL_1:.*]]: !stream.stream<i32>) -> !stream.stream<tuple<i32, i32>> {
// CHECK:           %[[VAL_2:.*]] = stream.combine(%[[VAL_0]], %[[VAL_1]]) {registers = [false]} : (!stream.stream<i32>, !stream.stream<i32>) -> !stream.stream<tuple<i32, i32>> {
// CHECK:           ^bb0(%[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i1):
// CHECK:             %[[VAL_6:.*]] = stream.pack %[[VAL_3]], %[[VAL_4]] : tuple<i32, i32>
// CHECK:             stream.yield %[[VAL_6]], %[[VAL_5]] : tuple<i32, i32>, i1
// CHECK:           }
// CHECK:           return %[[VAL_7:.*]] : !stream.stream<tuple<i32, i32>>
// CHECK:         }

func.func @combine(%in0: !stream.stream<i32>, %in1: !stream.stream<i32>) -> (!stream.stream<tuple<i32, i32>>) {
  %res = stream.combine(%in0, %in1) {registers = [0 : i1]}: (!stream.stream<i32>, !stream.stream<i32>) -> (!stream.stream<tuple<i32, i32>>) {
    ^0(%val0: i32, %val1: i32, %reg: i1):
    %0 = stream.pack %val0, %val1 : tuple<i32, i32>
    stream.yield %0, %reg : tuple<i32, i32>, i1
  }
  return %res : !stream.stream<tuple<i32, i32>>
}
