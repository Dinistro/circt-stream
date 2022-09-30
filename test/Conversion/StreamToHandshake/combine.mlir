// RUN: stream-opt %s --convert-stream-to-handshake | FileCheck %s

func.func @combine(%in0: !stream.stream<i32>, %in1: !stream.stream<i32>) -> (!stream.stream<i32>) {
  %res = stream.combine(%in0, %in1) : (!stream.stream<i32>, !stream.stream<i32>) -> (!stream.stream<i32>) {
  ^0(%val0: i32, %val1: i32):
  %0 = arith.addi %val0, %val1 : i32
    stream.yield %0 : i32
  }
  return %res : !stream.stream<i32>
}

// CHECK-LABEL:   handshake.func private @stream_combine(
// CHECK-SAME:      %[[VAL_0:.*]]: tuple<i32, i1>, %[[VAL_1:.*]]: none, %[[VAL_2:.*]]: tuple<i32, i1>, %[[VAL_3:.*]]: none, ...) -> (tuple<i32, i1>, none)
// CHECK:           %[[VAL_4:.*]]:2 = unpack %[[VAL_0]] : tuple<i32, i1>
// CHECK:           %[[VAL_5:.*]]:2 = unpack %[[VAL_2]] : tuple<i32, i1>
// CHECK:           %[[VAL_6:.*]] = arith.ori %[[VAL_4]]#1, %[[VAL_5]]#1 : i1
// CHECK:           %[[VAL_7:.*]]:6 = fork [6] %[[VAL_6]] : i1
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = cond_br %[[VAL_7]]#5, %[[VAL_4]]#0 : i32
// CHECK:           sink %[[VAL_8]] : i32
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = cond_br %[[VAL_7]]#4, %[[VAL_5]]#0 : i32
// CHECK:           sink %[[VAL_10]] : i32
// CHECK:           %[[VAL_12:.*]] = join %[[VAL_1]], %[[VAL_3]] : none, none
// CHECK:           %[[VAL_13:.*]], %[[VAL_14:.*]] = cond_br %[[VAL_7]]#3, %[[VAL_12]] : none
// CHECK:           sink %[[VAL_13]] : none
// CHECK:           %[[VAL_15:.*]] = merge %[[VAL_9]] : i32
// CHECK:           %[[VAL_16:.*]] = merge %[[VAL_11]] : i32
// CHECK:           %[[VAL_17:.*]] = arith.addi %[[VAL_15]], %[[VAL_16]] : i32
// CHECK:           %[[VAL_18:.*]] = source
// CHECK:           %[[VAL_19:.*]] = mux %[[VAL_7]]#2 {{\[}}%[[VAL_14]], %[[VAL_18]]] : i1, none
// CHECK:           %[[VAL_20:.*]] = source
// CHECK:           %[[VAL_21:.*]] = constant %[[VAL_20]] {value = 0 : i32} : i32
// CHECK:           %[[VAL_22:.*]] = mux %[[VAL_7]]#1 {{\[}}%[[VAL_17]], %[[VAL_21]]] : i1, i32
// CHECK:           %[[VAL_23:.*]] = pack %[[VAL_22]], %[[VAL_7]]#0 : tuple<i32, i1>
// CHECK:           return %[[VAL_23]], %[[VAL_19]] : tuple<i32, i1>, none
// CHECK:         }

// CHECK-LABEL:   handshake.func @combine(
// CHECK-SAME:      %[[VAL_0:.*]]: tuple<i32, i1>, %[[VAL_1:.*]]: none, %[[VAL_2:.*]]: tuple<i32, i1>, %[[VAL_3:.*]]: none, ...) -> (tuple<i32, i1>, none)
// CHECK:           %[[VAL_4:.*]]:2 = instance @stream_combine(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]]) : (tuple<i32, i1>, none, tuple<i32, i1>, none) -> (tuple<i32, i1>, none)
// CHECK:           return %[[VAL_4]]#0, %[[VAL_4]]#1 : tuple<i32, i1>, none
// CHECK:         }
