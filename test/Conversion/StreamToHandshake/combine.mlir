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
// CHECK-SAME:        %[[VAL_0:.*]]: tuple<i32, i1>, %[[VAL_1:.*]]: tuple<i32, i1>, ...) -> tuple<i32, i1>
// CHECK:           %[[VAL_2:.*]]:2 = unpack %[[VAL_0]] : tuple<i32, i1>
// CHECK:           %[[VAL_3:.*]]:2 = unpack %[[VAL_1]] : tuple<i32, i1>
// CHECK:           %[[VAL_4:.*]] = arith.ori %[[VAL_2]]#1, %[[VAL_3]]#1 : i1
// CHECK:           %[[VAL_5:.*]]:4 = fork [4] %[[VAL_4]] : i1
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = cond_br %[[VAL_5]]#3, %[[VAL_2]]#0 : i32
// CHECK:           sink %[[VAL_6]] : i32
// CHECK:           %[[VAL_8:.*]]:2 = fork [2] %[[VAL_7]] : i32
// CHECK:           %[[VAL_9:.*]], %[[VAL_10:.*]] = cond_br %[[VAL_5]]#2, %[[VAL_3]]#0 : i32
// CHECK:           sink %[[VAL_9]] : i32
// CHECK:           %[[VAL_11:.*]]:2 = fork [2] %[[VAL_10]] : i32
// CHECK:           %[[VAL_12:.*]] = join %[[VAL_8]]#1, %[[VAL_11]]#1 : i32, i32
// CHECK:           sink %[[VAL_12]] : none
// CHECK:           %[[VAL_13:.*]] = merge %[[VAL_8]]#0 : i32
// CHECK:           %[[VAL_14:.*]] = merge %[[VAL_11]]#0 : i32
// CHECK:           %[[VAL_15:.*]] = arith.addi %[[VAL_13]], %[[VAL_14]] : i32
// CHECK:           %[[VAL_16:.*]] = source
// CHECK:           %[[VAL_17:.*]] = constant %[[VAL_16]] {value = 0 : i32} : i32
// CHECK:           %[[VAL_18:.*]] = mux %[[VAL_5]]#1 {{\[}}%[[VAL_15]], %[[VAL_17]]] : i1, i32
// CHECK:           %[[VAL_19:.*]] = pack %[[VAL_18]], %[[VAL_5]]#0 : tuple<i32, i1>
// CHECK:           return %[[VAL_19]] : tuple<i32, i1>
// CHECK:         }

// CHECK-LABEL:   handshake.func @combine(
// CHECK-SAME:        %[[VAL_0:.*]]: tuple<i32, i1>, %[[VAL_1:.*]]: tuple<i32, i1>, ...) -> tuple<i32, i1>
// CHECK:           %[[VAL_2:.*]] = instance @stream_combine(%[[VAL_0]], %[[VAL_1]]) : (tuple<i32, i1>, tuple<i32, i1>) -> tuple<i32, i1>
// CHECK:           return %[[VAL_2]] : tuple<i32, i1>
// CHECK:         }
