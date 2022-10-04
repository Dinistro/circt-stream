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
// CHECK-SAME:                                         %[[VAL_0:.*]]: tuple<tuple<i32, i32>, i1>, ...) -> (tuple<i32, i1>, tuple<i32, i1>)
// CHECK:           %[[VAL_1:.*]]:2 = unpack %[[VAL_0]] : tuple<tuple<i32, i32>, i1>
// CHECK:           %[[VAL_2:.*]]:5 = fork [5] %[[VAL_1]]#1 : i1
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = cond_br %[[VAL_2]]#4, %[[VAL_1]]#0 : tuple<i32, i32>
// CHECK:           sink %[[VAL_3]] : tuple<i32, i32>
// CHECK:           %[[VAL_5:.*]]:2 = fork [2] %[[VAL_4]] : tuple<i32, i32>
// CHECK:           %[[VAL_6:.*]] = join %[[VAL_5]]#1 : tuple<i32, i32>
// CHECK:           sink %[[VAL_6]] : none
// CHECK:           %[[VAL_7:.*]] = merge %[[VAL_5]]#0 : tuple<i32, i32>
// CHECK:           %[[VAL_8:.*]]:2 = unpack %[[VAL_7]] : tuple<i32, i32>
// CHECK:           %[[VAL_9:.*]] = source
// CHECK:           %[[VAL_10:.*]] = constant %[[VAL_9]] {value = 0 : i32} : i32
// CHECK:           %[[VAL_11:.*]] = mux %[[VAL_2]]#3 {{\[}}%[[VAL_8]]#0, %[[VAL_10]]] : i1, i32
// CHECK:           %[[VAL_12:.*]] = pack %[[VAL_11]], %[[VAL_2]]#2 : tuple<i32, i1>
// CHECK:           %[[VAL_13:.*]] = source
// CHECK:           %[[VAL_14:.*]] = constant %[[VAL_13]] {value = 0 : i32} : i32
// CHECK:           %[[VAL_15:.*]] = mux %[[VAL_2]]#1 {{\[}}%[[VAL_8]]#1, %[[VAL_14]]] : i1, i32
// CHECK:           %[[VAL_16:.*]] = pack %[[VAL_15]], %[[VAL_2]]#0 : tuple<i32, i1>
// CHECK:           return %[[VAL_12]], %[[VAL_16]] : tuple<i32, i1>, tuple<i32, i1>
// CHECK:         }

// CHECK-LABEL:   handshake.func @split(
// CHECK-SAME:                          %[[VAL_0:.*]]: tuple<tuple<i32, i32>, i1>, ...) -> (tuple<i32, i1>, tuple<i32, i1>)
// CHECK:           %[[VAL_1:.*]]:2 = instance @stream_split(%[[VAL_0]]) : (tuple<tuple<i32, i32>, i1>) -> (tuple<i32, i1>, tuple<i32, i1>)
// CHECK:           return %[[VAL_1]]#0, %[[VAL_1]]#1 : tuple<i32, i1>, tuple<i32, i1>
// CHECK:         }
