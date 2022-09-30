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
// CHECK-SAME:                                          %[[VAL_0:.*]]: tuple<i512, i1>,
// CHECK-SAME:                                          %[[VAL_1:.*]]: none, ...) -> (tuple<i512, i1>, none)
// CHECK:           %[[VAL_2:.*]]:2 = fork [2] %[[VAL_1]] : none
// CHECK:           %[[VAL_3:.*]]:2 = fork [2] %[[VAL_0]] : tuple<i512, i1>
// CHECK:           %[[VAL_4:.*]]:2 = unpack %[[VAL_3]]#1 : tuple<i512, i1>
// CHECK:           %[[VAL_5:.*]]:5 = fork [5] %[[VAL_4]]#1 : i1
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = cond_br %[[VAL_5]]#4, %[[VAL_4]]#0 : i512
// CHECK:           sink %[[VAL_6]] : i512
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = cond_br %[[VAL_5]]#3, %[[VAL_2]]#1 : none
// CHECK:           sink %[[VAL_8]] : none
// CHECK:           %[[VAL_10:.*]] = merge %[[VAL_7]] : i512
// CHECK:           %[[VAL_11:.*]] = constant %[[VAL_9]] {value = 0 : i512} : i512
// CHECK:           %[[VAL_12:.*]] = arith.cmpi sgt, %[[VAL_10]], %[[VAL_11]] : i512
// CHECK:           %[[VAL_13:.*]], %[[VAL_14:.*]] = cond_br %[[VAL_5]]#1, %[[VAL_5]]#2 : i1
// CHECK:           sink %[[VAL_14]] : i1
// CHECK:           %[[VAL_15:.*]] = mux %[[VAL_5]]#0 {{\[}}%[[VAL_12]], %[[VAL_13]]] : i1, i1
// CHECK:           %[[VAL_16:.*]]:2 = fork [2] %[[VAL_15]] : i1
// CHECK:           %[[VAL_17:.*]], %[[VAL_18:.*]] = cond_br %[[VAL_16]]#1, %[[VAL_3]]#0 : tuple<i512, i1>
// CHECK:           sink %[[VAL_18]] : tuple<i512, i1>
// CHECK:           %[[VAL_19:.*]], %[[VAL_20:.*]] = cond_br %[[VAL_16]]#0, %[[VAL_2]]#0 : none
// CHECK:           sink %[[VAL_20]] : none
// CHECK:           return %[[VAL_17]], %[[VAL_19]] : tuple<i512, i1>, none
// CHECK:         }

// CHECK-LABEL:   handshake.func private @stream_map(
// CHECK-SAME:                                       %[[VAL_0:.*]]: tuple<i512, i1>,
// CHECK-SAME:                                       %[[VAL_1:.*]]: none, ...) -> (tuple<i512, i1>, none)
// CHECK:           %[[VAL_2:.*]] = source
// CHECK:           %[[VAL_3:.*]] = constant %[[VAL_2]] {value = 0 : i512} : i512
// CHECK:           %[[VAL_4:.*]] = mux %[[VAL_5:.*]]#2 {{\[}}%[[VAL_6:.*]], %[[VAL_3]]] : i1, i512
// CHECK:           %[[VAL_7:.*]]:2 = unpack %[[VAL_0]] : tuple<i512, i1>
// CHECK:           %[[VAL_5]]:5 = fork [5] %[[VAL_7]]#1 : i1
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = cond_br %[[VAL_5]]#4, %[[VAL_7]]#0 : i512
// CHECK:           sink %[[VAL_8]] : i512
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = cond_br %[[VAL_5]]#3, %[[VAL_1]] : none
// CHECK:           sink %[[VAL_10]] : none
// CHECK:           %[[VAL_12:.*]]:2 = fork [2] %[[VAL_11]] : none
// CHECK:           %[[VAL_13:.*]] = merge %[[VAL_9]] : i512
// CHECK:           %[[VAL_14:.*]] = constant %[[VAL_12]]#0 {value = 42 : i512} : i512
// CHECK:           %[[VAL_6]] = arith.addi %[[VAL_14]], %[[VAL_13]] : i512
// CHECK:           %[[VAL_15:.*]] = pack %[[VAL_4]], %[[VAL_5]]#1 : tuple<i512, i1>
// CHECK:           %[[VAL_16:.*]] = source
// CHECK:           %[[VAL_17:.*]] = mux %[[VAL_5]]#0 {{\[}}%[[VAL_12]]#1, %[[VAL_16]]] : i1, none
// CHECK:           return %[[VAL_15]], %[[VAL_17]] : tuple<i512, i1>, none
// CHECK:         }

// CHECK-LABEL:   handshake.func @combined(
// CHECK-SAME:                             %[[VAL_0:.*]]: tuple<i512, i1>,
// CHECK-SAME:                             %[[VAL_1:.*]]: none, ...) -> (tuple<i512, i1>, none)
// CHECK:           %[[VAL_2:.*]]:2 = instance @stream_map(%[[VAL_0]], %[[VAL_1]]) : (tuple<i512, i1>, none) -> (tuple<i512, i1>, none)
// CHECK:           %[[VAL_3:.*]]:2 = instance @stream_filter(%[[VAL_2]]#0, %[[VAL_2]]#1) : (tuple<i512, i1>, none) -> (tuple<i512, i1>, none)
// CHECK:           return %[[VAL_3]]#0, %[[VAL_3]]#1 : tuple<i512, i1>, none
// CHECK:         }
