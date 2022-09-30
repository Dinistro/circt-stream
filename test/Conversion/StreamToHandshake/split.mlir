// RUN: stream-opt %s --convert-stream-to-handshake | FileCheck %s

func.func @split(%in: !stream.stream<tuple<i32, i32>>) -> (!stream.stream<i32>, !stream.stream<i32>) {
  %res0, %res1 = stream.split(%in) : (!stream.stream<tuple<i32, i32>>) -> (!stream.stream<i32>, !stream.stream<i32>) {
  ^0(%val: tuple<i32, i32>):
    %0, %1 = stream.unpack %val : tuple<i32, i32>
    stream.yield %0, %1 : i32, i32
  }
  return %res0, %res1 : !stream.stream<i32>, !stream.stream<i32>
}

// CHECK-LABEL:   handshake.func private @stream_split(
// CHECK-SAME:                                         %[[VAL_0:.*]]: tuple<tuple<i32, i32>, i1>,
// CHECK-SAME:                                         %[[VAL_1:.*]]: none, ...) -> (tuple<i32, i1>, none, tuple<i32, i1>, none)
// CHECK:           %[[VAL_2:.*]]:2 = unpack %[[VAL_0]] : tuple<tuple<i32, i32>, i1>
// CHECK:           %[[VAL_3:.*]]:7 = fork [7] %[[VAL_2]]#1 : i1
// CHECK:           %[[VAL_4:.*]], %[[VAL_5:.*]] = cond_br %[[VAL_3]]#6, %[[VAL_2]]#0 : tuple<i32, i32>
// CHECK:           sink %[[VAL_4]] : tuple<i32, i32>
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = cond_br %[[VAL_3]]#5, %[[VAL_1]] : none
// CHECK:           sink %[[VAL_6]] : none
// CHECK:           %[[VAL_8:.*]] = merge %[[VAL_5]] : tuple<i32, i32>
// CHECK:           %[[VAL_9:.*]]:2 = unpack %[[VAL_8]] : tuple<i32, i32>
// CHECK:           %[[VAL_10:.*]] = source
// CHECK:           %[[VAL_11:.*]] = mux %[[VAL_3]]#4 {{\[}}%[[VAL_7]], %[[VAL_10]]] : i1, none
// CHECK:           %[[VAL_12:.*]]:2 = fork [2] %[[VAL_11]] : none
// CHECK:           %[[VAL_13:.*]] = source
// CHECK:           %[[VAL_14:.*]] = constant %[[VAL_13]] {value = 0 : i32} : i32
// CHECK:           %[[VAL_15:.*]] = mux %[[VAL_3]]#3 {{\[}}%[[VAL_9]]#0, %[[VAL_14]]] : i1, i32
// CHECK:           %[[VAL_16:.*]] = pack %[[VAL_15]], %[[VAL_3]]#2 : tuple<i32, i1>
// CHECK:           %[[VAL_17:.*]] = source
// CHECK:           %[[VAL_18:.*]] = constant %[[VAL_17]] {value = 0 : i32} : i32
// CHECK:           %[[VAL_19:.*]] = mux %[[VAL_3]]#1 {{\[}}%[[VAL_9]]#1, %[[VAL_18]]] : i1, i32
// CHECK:           %[[VAL_20:.*]] = pack %[[VAL_19]], %[[VAL_3]]#0 : tuple<i32, i1>
// CHECK:           return %[[VAL_16]], %[[VAL_12]]#0, %[[VAL_20]], %[[VAL_12]]#1 : tuple<i32, i1>, none, tuple<i32, i1>, none
// CHECK:         }

// CHECK-LABEL:   handshake.func @split(
// CHECK-SAME:                          %[[VAL_0:.*]]: tuple<tuple<i32, i32>, i1>,
// CHECK-SAME:                          %[[VAL_1:.*]]: none, ...) -> (tuple<i32, i1>, none, tuple<i32, i1>, none)
// CHECK:           %[[VAL_2:.*]]:4 = instance @stream_split(%[[VAL_0]], %[[VAL_1]]) : (tuple<tuple<i32, i32>, i1>, none) -> (tuple<i32, i1>, none, tuple<i32, i1>, none)
// CHECK:           return %[[VAL_2]]#0, %[[VAL_2]]#1, %[[VAL_2]]#2, %[[VAL_2]]#3 : tuple<i32, i1>, none, tuple<i32, i1>, none
// CHECK:         }
