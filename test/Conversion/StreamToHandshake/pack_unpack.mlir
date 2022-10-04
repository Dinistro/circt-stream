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
// CHECK-SAME:                                       %[[VAL_0:.*]]: tuple<tuple<i32, i64>, i1>, ...) -> tuple<tuple<i64, i32>, i1>
// CHECK:           %[[VAL_1:.*]] = source
// CHECK:           %[[VAL_2:.*]]:2 = fork [2] %[[VAL_1]] : none
// CHECK:           %[[VAL_3:.*]] = constant %[[VAL_2]]#1 {value = 0 : i64} : i64
// CHECK:           %[[VAL_4:.*]] = constant %[[VAL_2]]#0 {value = 0 : i32} : i32
// CHECK:           %[[VAL_5:.*]] = pack %[[VAL_3]], %[[VAL_4]] : tuple<i64, i32>
// CHECK:           %[[VAL_6:.*]] = mux %[[VAL_7:.*]]#1 {{\[}}%[[VAL_8:.*]], %[[VAL_5]]] : i1, tuple<i64, i32>
// CHECK:           %[[VAL_9:.*]]:2 = unpack %[[VAL_0]] : tuple<tuple<i32, i64>, i1>
// CHECK:           %[[VAL_7]]:3 = fork [3] %[[VAL_9]]#1 : i1
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = cond_br %[[VAL_7]]#2, %[[VAL_9]]#0 : tuple<i32, i64>
// CHECK:           sink %[[VAL_10]] : tuple<i32, i64>
// CHECK:           %[[VAL_12:.*]]:2 = fork [2] %[[VAL_11]] : tuple<i32, i64>
// CHECK:           %[[VAL_13:.*]] = join %[[VAL_12]]#1 : tuple<i32, i64>
// CHECK:           sink %[[VAL_13]] : none
// CHECK:           %[[VAL_14:.*]] = merge %[[VAL_12]]#0 : tuple<i32, i64>
// CHECK:           %[[VAL_15:.*]]:2 = unpack %[[VAL_14]] : tuple<i32, i64>
// CHECK:           %[[VAL_8]] = pack %[[VAL_15]]#1, %[[VAL_15]]#0 : tuple<i64, i32>
// CHECK:           %[[VAL_16:.*]] = pack %[[VAL_6]], %[[VAL_7]]#0 : tuple<tuple<i64, i32>, i1>
// CHECK:           return %[[VAL_16]] : tuple<tuple<i64, i32>, i1>
// CHECK:         }

// CHECK-LABEL:   handshake.func @map(
// CHECK-SAME:                        %[[VAL_0:.*]]: tuple<tuple<i32, i64>, i1>, ...) -> tuple<tuple<i64, i32>, i1>
// CHECK:           %[[VAL_1:.*]] = instance @stream_map(%[[VAL_0]]) : (tuple<tuple<i32, i64>, i1>) -> tuple<tuple<i64, i32>, i1>
// CHECK:           return %[[VAL_1]] : tuple<tuple<i64, i32>, i1>
// CHECK:         }
