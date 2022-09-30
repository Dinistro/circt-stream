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
// CHECK-SAME:                                          %[[VAL_0:.*]]: tuple<i64, i1>,
// CHECK-SAME:                                          %[[VAL_1:.*]]: none, ...) -> (tuple<i64, i1>, none)
// CHECK:           %[[VAL_2:.*]]:2 = unpack %[[VAL_0]] : tuple<i64, i1>
// CHECK:           %[[VAL_3:.*]]:6 = fork [6] %[[VAL_2]]#1 : i1
// CHECK:           %[[VAL_4:.*]] = source
// CHECK:           %[[VAL_5:.*]] = constant %[[VAL_4]] {value = 0 : i64} : i64
// CHECK:           %[[VAL_6:.*]] = mux %[[VAL_3]]#5 {{\[}}%[[VAL_7:.*]], %[[VAL_5]]] : i1, i64
// CHECK:           %[[VAL_8:.*]] = buffer [1] seq %[[VAL_6]] {initValues = [0]} : i64
// CHECK:           %[[VAL_9:.*]], %[[VAL_10:.*]] = cond_br %[[VAL_3]]#4, %[[VAL_8]] : i64
// CHECK:           %[[VAL_11:.*]]:2 = fork [2] %[[VAL_9]] : i64
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]] = cond_br %[[VAL_3]]#2, %[[VAL_3]]#3 : i1
// CHECK:           sink %[[VAL_13]] : i1
// CHECK:           %[[VAL_14:.*]], %[[VAL_15:.*]] = cond_br %[[VAL_3]]#1, %[[VAL_1]] : none
// CHECK:           sink %[[VAL_15]] : none
// CHECK:           %[[VAL_16:.*]]:2 = fork [2] %[[VAL_14]] : none
// CHECK:           %[[VAL_17:.*]], %[[VAL_18:.*]] = cond_br %[[VAL_3]]#0, %[[VAL_2]]#0 : i64
// CHECK:           sink %[[VAL_17]] : i64
// CHECK:           %[[VAL_19:.*]] = merge %[[VAL_10]] : i64
// CHECK:           %[[VAL_20:.*]] = merge %[[VAL_18]] : i64
// CHECK:           %[[VAL_7]] = arith.addi %[[VAL_19]], %[[VAL_20]] : i64
// CHECK:           %[[VAL_21:.*]] = constant %[[VAL_16]]#1 {value = false} : i1
// CHECK:           %[[VAL_22:.*]] = pack %[[VAL_11]]#1, %[[VAL_21]] : tuple<i64, i1>
// CHECK:           %[[VAL_23:.*]] = pack %[[VAL_11]]#0, %[[VAL_12]] : tuple<i64, i1>
// CHECK:           %[[VAL_24:.*]]:2 = fork [2] %[[VAL_23]] : tuple<i64, i1>
// CHECK:           %[[VAL_25:.*]] = buffer [2] seq %[[VAL_26:.*]]#2 {initValues = [1, 0]} : i1
// CHECK:           %[[VAL_26]]:3 = fork [3] %[[VAL_25]] : i1
// CHECK:           %[[VAL_27:.*]] = mux %[[VAL_26]]#1 {{\[}}%[[VAL_22]], %[[VAL_24]]#1] : i1, tuple<i64, i1>
// CHECK:           %[[VAL_28:.*]] = join %[[VAL_24]]#0 : tuple<i64, i1>
// CHECK:           %[[VAL_29:.*]] = mux %[[VAL_26]]#0 {{\[}}%[[VAL_16]]#0, %[[VAL_28]]] : i1, none
// CHECK:           return %[[VAL_27]], %[[VAL_29]] : tuple<i64, i1>, none
// CHECK:         }

// CHECK-LABEL:   handshake.func @reduce(
// CHECK-SAME:                           %[[VAL_0:.*]]: tuple<i64, i1>,
// CHECK-SAME:                           %[[VAL_1:.*]]: none, ...) -> (tuple<i64, i1>, none)
// CHECK:           %[[VAL_2:.*]]:2 = instance @stream_reduce(%[[VAL_0]], %[[VAL_1]]) : (tuple<i64, i1>, none) -> (tuple<i64, i1>, none)
// CHECK:           return %[[VAL_2]]#0, %[[VAL_2]]#1 : tuple<i64, i1>, none
// CHECK:         }
