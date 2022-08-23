// RUN: stream-opt %s --convert-stream-to-handshake --split-input-file | FileCheck %s

// CHECK-LABEL:   handshake.func private @stream_map(
// CHECK-SAME:      %[[VAL_0:.*]]: tuple<i32, i1>, %[[VAL_1:.*]]: none, %[[VAL_2:.*]]: none, ...) -> (tuple<i32, i1>, none, none)
// CHECK:           %[[VAL_3:.*]]:2 = unpack %[[VAL_0]] : tuple<i32, i1>
// CHECK:           %[[VAL_4:.*]] = buffer [1] seq %[[VAL_5:.*]] {initValues = [0]} : i32
// CHECK:           %[[VAL_6:.*]] = merge %[[VAL_3]]#0 : i32
// CHECK:           %[[VAL_5]] = merge %[[VAL_4]] : i32
// CHECK:           %[[VAL_7:.*]] = pack %[[VAL_6]], %[[VAL_3]]#1 : tuple<i32, i1>
// CHECK:           return %[[VAL_7]], %[[VAL_1]], %[[VAL_2]] : tuple<i32, i1>, none, none
// CHECK:         }


// CHECK-LABEL:   handshake.func @single_reg(
// CHECK-SAME:      %[[VAL_0:.*]]: tuple<i32, i1>, %[[VAL_1:.*]]: none, %[[VAL_2:.*]]: none, ...) -> none
// CHECK:           %[[VAL_3:.*]]:2 = fork [2] %[[VAL_2]] : none
// CHECK:           %[[VAL_4:.*]]:3 = instance @stream_map(%[[VAL_0]], %[[VAL_1]], %[[VAL_3]]#1) : (tuple<i32, i1>, none, none) -> (tuple<i32, i1>, none, none)

func.func @single_reg(%in: !stream.stream<i32>) {
  %res = stream.map(%in) {registers = [0 : i32]}: (!stream.stream<i32>) -> !stream.stream<i32> {
  ^0(%val : i32, %reg: i32):
    stream.yield %val, %reg : i32, i32
  }
  return
}

// -----

// CHECK-LABEL:   handshake.func private @stream_filter(
// CHECK-SAME:      %[[VAL_0:.*]]: tuple<i32, i1>, %[[VAL_1:.*]]: none, %[[VAL_2:.*]]: none, ...)
// CHECK:           %[[VAL_3:.*]]:2 = unpack %[[VAL_0]] : tuple<i32, i1>
// CHECK:           %[[VAL_4:.*]]:2 = fork [2] %[[VAL_3]]#1 : i1
// CHECK:           %[[VAL_5:.*]]:2 = fork [2] %[[VAL_3]]#0 : i32
// CHECK:           %[[VAL_6:.*]] = buffer [1] seq %[[VAL_7:.*]] {initValues = [0]} : i1
// CHECK:           %[[VAL_8:.*]] = buffer [1] seq %[[VAL_9:.*]]#1 {initValues = [-1]} : i1
// CHECK:           %[[VAL_10:.*]] = merge %[[VAL_5]]#0 : i32
// CHECK:           sink %[[VAL_10]] : i32
// CHECK:           %[[VAL_11:.*]] = merge %[[VAL_6]] : i1
// CHECK:           %[[VAL_9]]:2 = fork [2] %[[VAL_11]] : i1
// CHECK:           %[[VAL_7]] = merge %[[VAL_8]] : i1
// CHECK:           %[[VAL_12:.*]] = pack %[[VAL_5]]#1, %[[VAL_4]]#1 : tuple<i32, i1>
// CHECK:           %[[VAL_13:.*]] = arith.ori %[[VAL_9]]#0, %[[VAL_4]]#0 : i1
// CHECK:           %[[VAL_14:.*]]:2 = fork [2] %[[VAL_13]] : i1
// CHECK:           %[[VAL_15:.*]], %[[VAL_16:.*]] = cond_br %[[VAL_14]]#1, %[[VAL_12]] : tuple<i32, i1>
// CHECK:           sink %[[VAL_16]] : tuple<i32, i1>
// CHECK:           %[[VAL_17:.*]], %[[VAL_18:.*]] = cond_br %[[VAL_14]]#0, %[[VAL_1]] : none
// CHECK:           sink %[[VAL_18]] : none
// CHECK:           return %[[VAL_15]], %[[VAL_17]], %[[VAL_2]] : tuple<i32, i1>, none, none
// CHECK:         }

// CHECK-LABEL:   handshake.func @multiple_regs(
// CHECK-SAME:      %[[VAL_0:.*]]: tuple<i32, i1>, %[[VAL_1:.*]]: none, %[[VAL_2:.*]]: none, ...) -> none
// CHECK:           %[[VAL_3:.*]]:2 = fork [2] %[[VAL_2]] : none
// CHECK:           %[[VAL_4:.*]]:3 = instance @stream_filter(%[[VAL_0]], %[[VAL_1]], %[[VAL_3]]#1) : (tuple<i32, i1>, none, none) -> (tuple<i32, i1>, none, none)

func.func @multiple_regs(%in: !stream.stream<i32>) {
  %res = stream.filter(%in) {registers = [0 : i1, 1 : i1]}: (!stream.stream<i32>) -> !stream.stream<i32> {
  ^0(%val : i32, %reg0: i1, %reg1 : i1):
    stream.yield %reg0, %reg1, %reg0 : i1, i1, i1
  }
  return
}
