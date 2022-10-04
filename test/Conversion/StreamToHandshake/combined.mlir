// RUN: stream-opt %s --convert-stream-to-handshake | FileCheck %s

func.func @combined(%in: !stream.stream<i512>) -> !stream.stream<i512> {
  %tmp = stream.map(%in) : (!stream.stream<i512>) -> !stream.stream<i512> {
  ^0(%val : i512):
      %0 = arith.constant 42 : i512
      %r = arith.addi %0, %val : i512
      stream.yield %r : i512
  }
  %res = stream.filter(%tmp) : (!stream.stream<i512>) -> !stream.stream<i512> {
  ^bb0(%val: i512):
    %c0_i512 = arith.constant 0 : i512
    %0 = arith.cmpi sgt, %val, %c0_i512 : i512
    stream.yield %0 : i1
  }
  return %res : !stream.stream<i512>
}

// CHECK-LABEL:   handshake.func private @stream_filter(
// CHECK-SAME:                                          %[[VAL_0:.*]]: tuple<i512, i1>, ...) -> tuple<i512, i1>
// CHECK:           %[[VAL_1:.*]]:2 = fork [2] %[[VAL_0]] : tuple<i512, i1>
// CHECK:           %[[VAL_2:.*]]:2 = unpack %[[VAL_1]]#1 : tuple<i512, i1>
// CHECK:           %[[VAL_3:.*]]:4 = fork [4] %[[VAL_2]]#1 : i1
// CHECK:           %[[VAL_4:.*]], %[[VAL_5:.*]] = cond_br %[[VAL_3]]#3, %[[VAL_2]]#0 : i512
// CHECK:           sink %[[VAL_4]] : i512
// CHECK:           %[[VAL_6:.*]]:2 = fork [2] %[[VAL_5]] : i512
// CHECK:           %[[VAL_7:.*]] = join %[[VAL_6]]#1 : i512
// CHECK:           %[[VAL_8:.*]] = merge %[[VAL_6]]#0 : i512
// CHECK:           %[[VAL_9:.*]] = constant %[[VAL_7]] {value = 0 : i512} : i512
// CHECK:           %[[VAL_10:.*]] = arith.cmpi sgt, %[[VAL_8]], %[[VAL_9]] : i512
// CHECK:           %[[VAL_11:.*]], %[[VAL_12:.*]] = cond_br %[[VAL_3]]#1, %[[VAL_3]]#2 : i1
// CHECK:           sink %[[VAL_12]] : i1
// CHECK:           %[[VAL_13:.*]] = mux %[[VAL_3]]#0 {{\[}}%[[VAL_10]], %[[VAL_11]]] : i1, i1
// CHECK:           %[[VAL_14:.*]], %[[VAL_15:.*]] = cond_br %[[VAL_13]], %[[VAL_1]]#0 : tuple<i512, i1>
// CHECK:           sink %[[VAL_15]] : tuple<i512, i1>
// CHECK:           return %[[VAL_14]] : tuple<i512, i1>
// CHECK:         }

// CHECK-LABEL:   handshake.func private @stream_map(
// CHECK-SAME:                                       %[[VAL_0:.*]]: tuple<i512, i1>, ...) -> tuple<i512, i1>
// CHECK:           %[[VAL_1:.*]] = source
// CHECK:           %[[VAL_2:.*]] = constant %[[VAL_1]] {value = 0 : i512} : i512
// CHECK:           %[[VAL_3:.*]] = mux %[[VAL_4:.*]]#1 {{\[}}%[[VAL_5:.*]], %[[VAL_2]]] : i1, i512
// CHECK:           %[[VAL_6:.*]]:2 = unpack %[[VAL_0]] : tuple<i512, i1>
// CHECK:           %[[VAL_4]]:3 = fork [3] %[[VAL_6]]#1 : i1
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = cond_br %[[VAL_4]]#2, %[[VAL_6]]#0 : i512
// CHECK:           sink %[[VAL_7]] : i512
// CHECK:           %[[VAL_9:.*]]:2 = fork [2] %[[VAL_8]] : i512
// CHECK:           %[[VAL_10:.*]] = join %[[VAL_9]]#1 : i512
// CHECK:           %[[VAL_11:.*]] = merge %[[VAL_9]]#0 : i512
// CHECK:           %[[VAL_12:.*]] = constant %[[VAL_10]] {value = 42 : i512} : i512
// CHECK:           %[[VAL_5]] = arith.addi %[[VAL_12]], %[[VAL_11]] : i512
// CHECK:           %[[VAL_13:.*]] = pack %[[VAL_3]], %[[VAL_4]]#0 : tuple<i512, i1>
// CHECK:           return %[[VAL_13]] : tuple<i512, i1>
// CHECK:         }

// CHECK-LABEL:   handshake.func @combined(
// CHECK-SAME:                             %[[VAL_0:.*]]: tuple<i512, i1>, ...) -> tuple<i512, i1>
// CHECK:           %[[VAL_1:.*]] = instance @stream_map(%[[VAL_0]]) : (tuple<i512, i1>) -> tuple<i512, i1>
// CHECK:           %[[VAL_2:.*]] = instance @stream_filter(%[[VAL_1]]) : (tuple<i512, i1>) -> tuple<i512, i1>
// CHECK:           return %[[VAL_2]] : tuple<i512, i1>
// CHECK:         }
