// RUN: stream-opt %s --convert-stream-to-handshake | FileCheck %s
func.func @reduce(%in: !stream.stream<i64>) -> !stream.stream<i64> {
  %res = stream.reduce(%in) {initValue = 0 : i64}: (!stream.stream<i64>) -> !stream.stream<i64> {
  ^0(%acc: i64, %val: i64):
    %r = arith.addi %acc, %val : i64
    stream.yield %r : i64
  }
  return %res : !stream.stream<i64>
}

// CHECK-LABEL:   handshake.func private @stream_reduce(
// CHECK-SAME:                                          %[[VAL_0:.*]]: tuple<i64, i1>, ...) -> tuple<i64, i1>
// CHECK:           %[[VAL_1:.*]]:2 = unpack %[[VAL_0]] : tuple<i64, i1>
// CHECK:           %[[VAL_2:.*]]:5 = fork [5] %[[VAL_1]]#1 : i1
// CHECK:           %[[VAL_3:.*]] = source
// CHECK:           %[[VAL_4:.*]] = constant %[[VAL_3]] {value = 0 : i64} : i64
// CHECK:           %[[VAL_5:.*]] = mux %[[VAL_2]]#4 {{\[}}%[[VAL_6:.*]], %[[VAL_4]]] : i1, i64
// CHECK:           %[[VAL_7:.*]] = buffer [1] seq %[[VAL_5]] {initValues = [0]} : i64
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = cond_br %[[VAL_2]]#3, %[[VAL_7]] : i64
// CHECK:           %[[VAL_10:.*]]:2 = fork [2] %[[VAL_9]] : i64
// CHECK:           %[[VAL_11:.*]]:3 = fork [3] %[[VAL_8]] : i64
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]] = cond_br %[[VAL_2]]#1, %[[VAL_2]]#2 : i1
// CHECK:           sink %[[VAL_13]] : i1
// CHECK:           %[[VAL_14:.*]], %[[VAL_15:.*]] = cond_br %[[VAL_2]]#0, %[[VAL_1]]#0 : i64
// CHECK:           sink %[[VAL_14]] : i64
// CHECK:           %[[VAL_16:.*]] = join %[[VAL_10]]#1 : i64
// CHECK:           sink %[[VAL_16]] : none
// CHECK:           %[[VAL_17:.*]] = merge %[[VAL_10]]#0 : i64
// CHECK:           %[[VAL_18:.*]] = merge %[[VAL_15]] : i64
// CHECK:           %[[VAL_6]] = arith.addi %[[VAL_17]], %[[VAL_18]] : i64
// CHECK:           %[[VAL_19:.*]] = join %[[VAL_11]]#2 : i64
// CHECK:           %[[VAL_20:.*]] = constant %[[VAL_19]] {value = false} : i1
// CHECK:           %[[VAL_21:.*]] = pack %[[VAL_11]]#1, %[[VAL_20]] : tuple<i64, i1>
// CHECK:           %[[VAL_22:.*]] = pack %[[VAL_11]]#0, %[[VAL_12]] : tuple<i64, i1>
// CHECK:           %[[VAL_23:.*]] = buffer [2] seq %[[VAL_24:.*]]#1 {initValues = [1, 0]} : i1
// CHECK:           %[[VAL_24]]:2 = fork [2] %[[VAL_23]] : i1
// CHECK:           %[[VAL_25:.*]] = mux %[[VAL_24]]#0 {{\[}}%[[VAL_21]], %[[VAL_22]]] : i1, tuple<i64, i1>
// CHECK:           return %[[VAL_25]] : tuple<i64, i1>
// CHECK:         }

// CHECK-LABEL:   handshake.func @reduce(
// CHECK-SAME:                           %[[VAL_0:.*]]: tuple<i64, i1>, ...) -> tuple<i64, i1>
// CHECK:           %[[VAL_1:.*]] = instance @stream_reduce(%[[VAL_0]]) : (tuple<i64, i1>) -> tuple<i64, i1>
// CHECK:           return %[[VAL_1]] : tuple<i64, i1>
// CHECK:         }
