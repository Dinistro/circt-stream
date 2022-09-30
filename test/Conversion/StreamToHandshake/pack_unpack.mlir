// RUN: stream-opt %s --convert-stream-to-handshake | FileCheck %s

func.func @map(%in: !stream.stream<tuple<i32, i64>>) -> !stream.stream<tuple<i64, i32>> {
  %res = stream.map(%in) : (!stream.stream<tuple<i32, i64>>) -> !stream.stream<tuple<i64, i32>> {
  ^0(%val : tuple<i32, i64>):
    %a, %b = stream.unpack %val : tuple<i32, i64>
    %r = stream.pack %b, %a : tuple<i64, i32>
    stream.yield %r : tuple<i64, i32>
  }
  return %res : !stream.stream<tuple<i64, i32>>
}

// CHECK-LABEL:   handshake.func private @stream_map(
// CHECK-SAME:                                       %[[VAL_0:.*]]: tuple<tuple<i32, i64>, i1>,
// CHECK-SAME:                                       %[[VAL_1:.*]]: none, ...) -> (tuple<tuple<i64, i32>, i1>, none)
// CHECK:           %[[VAL_2:.*]] = source
// CHECK:           %[[VAL_3:.*]]:2 = fork [2] %[[VAL_2]] : none
// CHECK:           %[[VAL_4:.*]] = constant %[[VAL_3]]#1 {value = 0 : i64} : i64
// CHECK:           %[[VAL_5:.*]] = constant %[[VAL_3]]#0 {value = 0 : i32} : i32
// CHECK:           %[[VAL_6:.*]] = pack %[[VAL_4]], %[[VAL_5]] : tuple<i64, i32>
// CHECK:           %[[VAL_7:.*]] = mux %[[VAL_8:.*]]#2 {{\[}}%[[VAL_9:.*]], %[[VAL_6]]] : i1, tuple<i64, i32>
// CHECK:           %[[VAL_10:.*]]:2 = unpack %[[VAL_0]] : tuple<tuple<i32, i64>, i1>
// CHECK:           %[[VAL_8]]:5 = fork [5] %[[VAL_10]]#1 : i1
// CHECK:           %[[VAL_11:.*]], %[[VAL_12:.*]] = cond_br %[[VAL_8]]#4, %[[VAL_10]]#0 : tuple<i32, i64>
// CHECK:           sink %[[VAL_11]] : tuple<i32, i64>
// CHECK:           %[[VAL_13:.*]], %[[VAL_14:.*]] = cond_br %[[VAL_8]]#3, %[[VAL_1]] : none
// CHECK:           sink %[[VAL_13]] : none
// CHECK:           %[[VAL_15:.*]] = merge %[[VAL_12]] : tuple<i32, i64>
// CHECK:           %[[VAL_16:.*]]:2 = unpack %[[VAL_15]] : tuple<i32, i64>
// CHECK:           %[[VAL_9]] = pack %[[VAL_16]]#1, %[[VAL_16]]#0 : tuple<i64, i32>
// CHECK:           %[[VAL_17:.*]] = pack %[[VAL_7]], %[[VAL_8]]#1 : tuple<tuple<i64, i32>, i1>
// CHECK:           %[[VAL_18:.*]] = source
// CHECK:           %[[VAL_19:.*]] = mux %[[VAL_8]]#0 {{\[}}%[[VAL_14]], %[[VAL_18]]] : i1, none
// CHECK:           return %[[VAL_17]], %[[VAL_19]] : tuple<tuple<i64, i32>, i1>, none
// CHECK:         }

// CHECK-LABEL:   handshake.func @map(
// CHECK-SAME:                        %[[VAL_0:.*]]: tuple<tuple<i32, i64>, i1>,
// CHECK-SAME:                        %[[VAL_1:.*]]: none, ...) -> (tuple<tuple<i64, i32>, i1>, none)
// CHECK:           %[[VAL_2:.*]]:2 = instance @stream_map(%[[VAL_0]], %[[VAL_1]]) : (tuple<tuple<i32, i64>, i1>, none) -> (tuple<tuple<i64, i32>, i1>, none)
// CHECK:           return %[[VAL_2]]#0, %[[VAL_2]]#1 : tuple<tuple<i64, i32>, i1>, none
// CHECK:         }
