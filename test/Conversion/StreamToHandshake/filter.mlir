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
// CHECK-SAME:                                          %[[VAL_0:.*]]: tuple<i32, i1>, ...) -> tuple<i32, i1>
// CHECK:           %[[VAL_1:.*]]:2 = fork [2] %[[VAL_0]] : tuple<i32, i1>
// CHECK:           %[[VAL_2:.*]]:2 = unpack %[[VAL_1]]#1 : tuple<i32, i1>
// CHECK:           %[[VAL_3:.*]]:4 = fork [4] %[[VAL_2]]#1 : i1
// CHECK:           %[[VAL_4:.*]], %[[VAL_5:.*]] = cond_br %[[VAL_3]]#3, %[[VAL_2]]#0 : i32
// CHECK:           sink %[[VAL_4]] : i32
// CHECK:           %[[VAL_6:.*]]:2 = fork [2] %[[VAL_5]] : i32
// CHECK:           %[[VAL_7:.*]] = join %[[VAL_6]]#1 : i32
// CHECK:           %[[VAL_8:.*]] = merge %[[VAL_6]]#0 : i32
// CHECK:           %[[VAL_9:.*]] = constant %[[VAL_7]] {value = 0 : i32} : i32
// CHECK:           %[[VAL_10:.*]] = arith.cmpi sgt, %[[VAL_8]], %[[VAL_9]] : i32
// CHECK:           %[[VAL_11:.*]], %[[VAL_12:.*]] = cond_br %[[VAL_3]]#1, %[[VAL_3]]#2 : i1
// CHECK:           sink %[[VAL_12]] : i1
// CHECK:           %[[VAL_13:.*]] = mux %[[VAL_3]]#0 {{\[}}%[[VAL_10]], %[[VAL_11]]] : i1, i1
// CHECK:           %[[VAL_14:.*]], %[[VAL_15:.*]] = cond_br %[[VAL_13]], %[[VAL_1]]#0 : tuple<i32, i1>
// CHECK:           sink %[[VAL_15]] : tuple<i32, i1>
// CHECK:           return %[[VAL_14]] : tuple<i32, i1>
// CHECK:         }

// CHECK-LABEL:   handshake.func @filter(
// CHECK-SAME:                           %[[VAL_0:.*]]: tuple<i32, i1>, ...) -> tuple<i32, i1>
// CHECK:           %[[VAL_1:.*]] = instance @stream_filter(%[[VAL_0]]) : (tuple<i32, i1>) -> tuple<i32, i1>
// CHECK:           return %[[VAL_1]] : tuple<i32, i1>
// CHECK:         }
