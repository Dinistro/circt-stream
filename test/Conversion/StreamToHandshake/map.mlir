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
// CHECK-SAME:                                       %[[VAL_0:.*]]: tuple<i32, i1>, ...) -> tuple<i32, i1>
// CHECK:           %[[VAL_1:.*]] = source
// CHECK:           %[[VAL_2:.*]] = constant %[[VAL_1]] {value = 0 : i32} : i32
// CHECK:           %[[VAL_3:.*]] = mux %[[VAL_4:.*]]#1 {{\[}}%[[VAL_5:.*]], %[[VAL_2]]] : i1, i32
// CHECK:           %[[VAL_6:.*]]:2 = unpack %[[VAL_0]] : tuple<i32, i1>
// CHECK:           %[[VAL_4]]:3 = fork [3] %[[VAL_6]]#1 : i1
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = cond_br %[[VAL_4]]#2, %[[VAL_6]]#0 : i32
// CHECK:           sink %[[VAL_7]] : i32
// CHECK:           %[[VAL_9:.*]]:2 = fork [2] %[[VAL_8]] : i32
// CHECK:           %[[VAL_10:.*]] = join %[[VAL_9]]#1 : i32
// CHECK:           %[[VAL_11:.*]] = merge %[[VAL_9]]#0 : i32
// CHECK:           %[[VAL_12:.*]] = constant %[[VAL_10]] {value = 1 : i32} : i32
// CHECK:           %[[VAL_5]] = arith.addi %[[VAL_12]], %[[VAL_11]] : i32
// CHECK:           %[[VAL_13:.*]] = pack %[[VAL_3]], %[[VAL_4]]#0 : tuple<i32, i1>
// CHECK:           return %[[VAL_13]] : tuple<i32, i1>
// CHECK:         }

// CHECK-LABEL:   handshake.func @map(
// CHECK-SAME:                        %[[VAL_0:.*]]: tuple<i32, i1>, ...) -> tuple<i32, i1>
// CHECK:           %[[VAL_1:.*]] = instance @stream_map(%[[VAL_0]]) : (tuple<i32, i1>) -> tuple<i32, i1>
// CHECK:           return %[[VAL_1]] : tuple<i32, i1>
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

// CHECK:  handshake.func private @[[LABEL:.*]](%{{.*}}: tuple<i32, i1>, ...) -> tuple<i32, i1>
// CHECK:   handshake.func @map_multi_block(%{{.*}}: tuple<i32, i1>, ...) -> tuple<i32, i1>
