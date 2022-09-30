// RUN: stream-opt %s --convert-stream-to-handshake | FileCheck %s

func.func @filter(%in: !stream.stream<i32>) -> !stream.stream<i32> {
  %out = stream.filter(%in) : (!stream.stream<i32>) -> !stream.stream<i32> {
  ^bb0(%val: i32):
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi sgt, %val, %c0_i32 : i32
    stream.yield %0 : i1
  }
  return %out : !stream.stream<i32>
}

// CHECK-LABEL:   handshake.func private @stream_filter(
// CHECK-SAME:                                          %[[VAL_0:.*]]: tuple<i32, i1>,
// CHECK-SAME:                                          %[[VAL_1:.*]]: none, ...) -> (tuple<i32, i1>, none)
// CHECK:           %[[VAL_2:.*]]:2 = fork [2] %[[VAL_1]] : none
// CHECK:           %[[VAL_3:.*]]:2 = fork [2] %[[VAL_0]] : tuple<i32, i1>
// CHECK:           %[[VAL_4:.*]]:2 = unpack %[[VAL_3]]#1 : tuple<i32, i1>
// CHECK:           %[[VAL_5:.*]]:5 = fork [5] %[[VAL_4]]#1 : i1
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = cond_br %[[VAL_5]]#4, %[[VAL_4]]#0 : i32
// CHECK:           sink %[[VAL_6]] : i32
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = cond_br %[[VAL_5]]#3, %[[VAL_2]]#1 : none
// CHECK:           sink %[[VAL_8]] : none
// CHECK:           %[[VAL_10:.*]] = merge %[[VAL_7]] : i32
// CHECK:           %[[VAL_11:.*]] = constant %[[VAL_9]] {value = 0 : i32} : i32
// CHECK:           %[[VAL_12:.*]] = arith.cmpi sgt, %[[VAL_10]], %[[VAL_11]] : i32
// CHECK:           %[[VAL_13:.*]], %[[VAL_14:.*]] = cond_br %[[VAL_5]]#1, %[[VAL_5]]#2 : i1
// CHECK:           sink %[[VAL_14]] : i1
// CHECK:           %[[VAL_15:.*]] = mux %[[VAL_5]]#0 {{\[}}%[[VAL_12]], %[[VAL_13]]] : i1, i1
// CHECK:           %[[VAL_16:.*]]:2 = fork [2] %[[VAL_15]] : i1
// CHECK:           %[[VAL_17:.*]], %[[VAL_18:.*]] = cond_br %[[VAL_16]]#1, %[[VAL_3]]#0 : tuple<i32, i1>
// CHECK:           sink %[[VAL_18]] : tuple<i32, i1>
// CHECK:           %[[VAL_19:.*]], %[[VAL_20:.*]] = cond_br %[[VAL_16]]#0, %[[VAL_2]]#0 : none
// CHECK:           sink %[[VAL_20]] : none
// CHECK:           return %[[VAL_17]], %[[VAL_19]] : tuple<i32, i1>, none
// CHECK:         }

// CHECK-LABEL:   handshake.func @filter(
// CHECK-SAME:                           %[[VAL_0:.*]]: tuple<i32, i1>,
// CHECK-SAME:                           %[[VAL_1:.*]]: none, ...) -> (tuple<i32, i1>, none)
// CHECK:           %[[VAL_2:.*]]:2 = instance @stream_filter(%[[VAL_0]], %[[VAL_1]]) : (tuple<i32, i1>, none) -> (tuple<i32, i1>, none)
// CHECK:           return %[[VAL_2]]#0, %[[VAL_2]]#1 : tuple<i32, i1>, none
// CHECK:         }
