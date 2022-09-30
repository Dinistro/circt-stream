// RUN: stream-opt %s --convert-stream-to-handshake --split-input-file | FileCheck %s

func.func @map(%in: !stream.stream<i32>) -> !stream.stream<i32> {
  %res = stream.map(%in) : (!stream.stream<i32>) -> !stream.stream<i32> {
  ^0(%val : i32):
    %0 = arith.constant 1 : i32
    %r = arith.addi %0, %val : i32
    stream.yield %r : i32
  }
  return %res : !stream.stream<i32>
}

// CHECK-LABEL:   handshake.func private @stream_map(
// CHECK-SAME:                                       %[[VAL_0:.*]]: tuple<i32, i1>,
// CHECK-SAME:                                       %[[VAL_1:.*]]: none, ...) -> (tuple<i32, i1>, none)
// CHECK:           %[[VAL_2:.*]] = source
// CHECK:           %[[VAL_3:.*]] = constant %[[VAL_2]] {value = 0 : i32} : i32
// CHECK:           %[[VAL_4:.*]] = mux %[[VAL_5:.*]]#2 {{\[}}%[[VAL_6:.*]], %[[VAL_3]]] : i1, i32
// CHECK:           %[[VAL_7:.*]]:2 = unpack %[[VAL_0]] : tuple<i32, i1>
// CHECK:           %[[VAL_5]]:5 = fork [5] %[[VAL_7]]#1 : i1
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = cond_br %[[VAL_5]]#4, %[[VAL_7]]#0 : i32
// CHECK:           sink %[[VAL_8]] : i32
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = cond_br %[[VAL_5]]#3, %[[VAL_1]] : none
// CHECK:           sink %[[VAL_10]] : none
// CHECK:           %[[VAL_12:.*]]:2 = fork [2] %[[VAL_11]] : none
// CHECK:           %[[VAL_13:.*]] = merge %[[VAL_9]] : i32
// CHECK:           %[[VAL_14:.*]] = constant %[[VAL_12]]#0 {value = 1 : i32} : i32
// CHECK:           %[[VAL_6]] = arith.addi %[[VAL_14]], %[[VAL_13]] : i32
// CHECK:           %[[VAL_15:.*]] = pack %[[VAL_4]], %[[VAL_5]]#1 : tuple<i32, i1>
// CHECK:           %[[VAL_16:.*]] = source
// CHECK:           %[[VAL_17:.*]] = mux %[[VAL_5]]#0 {{\[}}%[[VAL_12]]#1, %[[VAL_16]]] : i1, none
// CHECK:           return %[[VAL_15]], %[[VAL_17]] : tuple<i32, i1>, none
// CHECK:         }

// CHECK-LABEL:   handshake.func @map(
// CHECK-SAME:                        %[[VAL_0:.*]]: tuple<i32, i1>,
// CHECK-SAME:                        %[[VAL_1:.*]]: none, ...) -> (tuple<i32, i1>, none)
// CHECK:           %[[VAL_2:.*]]:2 = instance @stream_map(%[[VAL_0]], %[[VAL_1]]) : (tuple<i32, i1>, none) -> (tuple<i32, i1>, none)
// CHECK:           return %[[VAL_2]]#0, %[[VAL_2]]#1 : tuple<i32, i1>, none
// CHECK:         }

// -----

func.func @map_multi_block(%in: !stream.stream<i32>) -> !stream.stream<i32> {
  %res = stream.map(%in) : (!stream.stream<i32>) -> !stream.stream<i32> {
  ^0(%val : i32):
    %c0 = arith.constant 0 : i32
    %cond = arith.cmpi eq, %c0, %val : i32 
    cf.cond_br %cond, ^1, ^2(%val: i32)
  ^1:
    %0 = arith.constant 1 : i32
    %r = arith.addi %0, %val : i32
    cf.br ^2(%r: i32)
  ^2(%out: i32):
    stream.yield %out : i32
  }
  return %res : !stream.stream<i32>
}

// CHECK:  handshake.func private @[[LABEL:.*]](%{{.*}}: tuple<i32, i1>, %{{.*}}: none, ...) -> (tuple<i32, i1>, none)
// CHECK:   handshake.func @map_multi_block(%{{.*}}: tuple<i32, i1>, %{{.*}}: none, ...) -> (tuple<i32, i1>, none)
