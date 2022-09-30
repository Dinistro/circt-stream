// RUN: stream-opt %s --convert-stream-to-handshake --split-input-file | FileCheck %s

// CHECK-LABEL:   handshake.func private @stream_map(
// CHECK-SAME:                                       %[[VAL_0:.*]]: tuple<i32, i1>,
// CHECK-SAME:                                       %[[VAL_1:.*]]: none, ...) -> (tuple<i32, i1>, none)
// CHECK:           %[[VAL_2:.*]] = source
// CHECK:           %[[VAL_3:.*]] = constant %[[VAL_2]] {value = 0 : i32} : i32
// CHECK:           %[[VAL_4:.*]] = mux %[[VAL_5:.*]]#2 {{\[}}%[[VAL_6:.*]], %[[VAL_3]]] : i1, i32
// CHECK:           %[[VAL_7:.*]]:2 = unpack %[[VAL_0]] : tuple<i32, i1>
// CHECK:           %[[VAL_5]]:7 = fork [7] %[[VAL_7]]#1 : i1
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = cond_br %[[VAL_5]]#6, %[[VAL_7]]#0 : i32
// CHECK:           sink %[[VAL_8]] : i32
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = cond_br %[[VAL_5]]#5, %[[VAL_1]] : none
// CHECK:           sink %[[VAL_10]] : none
// CHECK:           %[[VAL_12:.*]] = source
// CHECK:           %[[VAL_13:.*]] = constant %[[VAL_12]] {value = 0 : i32} : i32
// CHECK:           %[[VAL_14:.*]] = mux %[[VAL_5]]#4 {{\[}}%[[VAL_15:.*]], %[[VAL_13]]] : i1, i32
// CHECK:           %[[VAL_16:.*]] = buffer [1] seq %[[VAL_14]] {initValues = [0]} : i32
// CHECK:           %[[VAL_17:.*]], %[[VAL_18:.*]] = cond_br %[[VAL_5]]#3, %[[VAL_16]] : i32
// CHECK:           sink %[[VAL_17]] : i32
// CHECK:           %[[VAL_6]] = merge %[[VAL_9]] : i32
// CHECK:           %[[VAL_15]] = merge %[[VAL_18]] : i32
// CHECK:           %[[VAL_19:.*]] = pack %[[VAL_4]], %[[VAL_5]]#1 : tuple<i32, i1>
// CHECK:           %[[VAL_20:.*]] = source
// CHECK:           %[[VAL_21:.*]] = mux %[[VAL_5]]#0 {{\[}}%[[VAL_11]], %[[VAL_20]]] : i1, none
// CHECK:           return %[[VAL_19]], %[[VAL_21]] : tuple<i32, i1>, none
// CHECK:         }

// CHECK-LABEL:   handshake.func @single_reg(
// CHECK-SAME:                               %[[VAL_0:.*]]: tuple<i32, i1>,
// CHECK-SAME:                               %[[VAL_1:.*]]: none, ...) -> (tuple<i32, i1>, none)
// CHECK:           %[[VAL_2:.*]]:2 = instance @stream_map(%[[VAL_0]], %[[VAL_1]]) : (tuple<i32, i1>, none) -> (tuple<i32, i1>, none)
// CHECK:           return %[[VAL_2]]#0, %[[VAL_2]]#1 : tuple<i32, i1>, none
// CHECK:         }

func.func @single_reg(%in: !stream.stream<i32>) -> !stream.stream<i32> {
  %res = stream.map(%in) {registers = [0 : i32]}: (!stream.stream<i32>) -> !stream.stream<i32> {
  ^0(%val : i32, %reg: i32):
    stream.yield %val, %reg : i32, i32
  }
  return %res: !stream.stream<i32>
}

// -----

// CHECK-LABEL:   handshake.func private @stream_filter(
// CHECK-SAME:                                          %[[VAL_0:.*]]: tuple<i32, i1>,
// CHECK-SAME:                                          %[[VAL_1:.*]]: none, ...) -> (tuple<i32, i1>, none)
// CHECK:           %[[VAL_2:.*]]:2 = fork [2] %[[VAL_1]] : none
// CHECK:           %[[VAL_3:.*]]:2 = fork [2] %[[VAL_0]] : tuple<i32, i1>
// CHECK:           %[[VAL_4:.*]]:2 = unpack %[[VAL_3]]#1 : tuple<i32, i1>
// CHECK:           %[[VAL_5:.*]]:9 = fork [9] %[[VAL_4]]#1 : i1
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = cond_br %[[VAL_5]]#8, %[[VAL_4]]#0 : i32
// CHECK:           sink %[[VAL_6]] : i32
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = cond_br %[[VAL_5]]#7, %[[VAL_2]]#1 : none
// CHECK:           sink %[[VAL_9]] : none
// CHECK:           sink %[[VAL_8]] : none
// CHECK:           %[[VAL_10:.*]] = source
// CHECK:           %[[VAL_11:.*]] = constant %[[VAL_10]] {value = false} : i1
// CHECK:           %[[VAL_12:.*]] = mux %[[VAL_5]]#6 {{\[}}%[[VAL_13:.*]], %[[VAL_11]]] : i1, i1
// CHECK:           %[[VAL_14:.*]] = buffer [1] seq %[[VAL_12]] {initValues = [0]} : i1
// CHECK:           %[[VAL_15:.*]], %[[VAL_16:.*]] = cond_br %[[VAL_5]]#5, %[[VAL_14]] : i1
// CHECK:           sink %[[VAL_15]] : i1
// CHECK:           %[[VAL_17:.*]] = source
// CHECK:           %[[VAL_18:.*]] = constant %[[VAL_17]] {value = true} : i1
// CHECK:           %[[VAL_19:.*]] = mux %[[VAL_5]]#4 {{\[}}%[[VAL_20:.*]]#1, %[[VAL_18]]] : i1, i1
// CHECK:           %[[VAL_21:.*]] = buffer [1] seq %[[VAL_19]] {initValues = [-1]} : i1
// CHECK:           %[[VAL_22:.*]], %[[VAL_23:.*]] = cond_br %[[VAL_5]]#3, %[[VAL_21]] : i1
// CHECK:           sink %[[VAL_22]] : i1
// CHECK:           %[[VAL_24:.*]] = merge %[[VAL_7]] : i32
// CHECK:           sink %[[VAL_24]] : i32
// CHECK:           %[[VAL_25:.*]] = merge %[[VAL_16]] : i1
// CHECK:           %[[VAL_20]]:2 = fork [2] %[[VAL_25]] : i1
// CHECK:           %[[VAL_13]] = merge %[[VAL_23]] : i1
// CHECK:           %[[VAL_26:.*]], %[[VAL_27:.*]] = cond_br %[[VAL_5]]#1, %[[VAL_5]]#2 : i1
// CHECK:           sink %[[VAL_27]] : i1
// CHECK:           %[[VAL_28:.*]] = mux %[[VAL_5]]#0 {{\[}}%[[VAL_20]]#0, %[[VAL_26]]] : i1, i1
// CHECK:           %[[VAL_29:.*]]:2 = fork [2] %[[VAL_28]] : i1
// CHECK:           %[[VAL_30:.*]], %[[VAL_31:.*]] = cond_br %[[VAL_29]]#1, %[[VAL_3]]#0 : tuple<i32, i1>
// CHECK:           sink %[[VAL_31]] : tuple<i32, i1>
// CHECK:           %[[VAL_32:.*]], %[[VAL_33:.*]] = cond_br %[[VAL_29]]#0, %[[VAL_2]]#0 : none
// CHECK:           sink %[[VAL_33]] : none
// CHECK:           return %[[VAL_30]], %[[VAL_32]] : tuple<i32, i1>, none
// CHECK:         }

// CHECK-LABEL:   handshake.func @multiple_regs(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tuple<i32, i1>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: none, ...) -> (tuple<i32, i1>, none)
// CHECK:           %[[VAL_2:.*]]:2 = instance @stream_filter(%[[VAL_0]], %[[VAL_1]]) : (tuple<i32, i1>, none) -> (tuple<i32, i1>, none)
// CHECK:           return %[[VAL_2]]#0, %[[VAL_2]]#1 : tuple<i32, i1>, none
// CHECK:         }


func.func @multiple_regs(%in: !stream.stream<i32>) -> !stream.stream<i32> {
  %res = stream.filter(%in) {registers = [0 : i1, 1 : i1]}: (!stream.stream<i32>) -> !stream.stream<i32> {
  ^0(%val : i32, %reg0: i1, %reg1 : i1):
    stream.yield %reg0, %reg1, %reg0 : i1, i1, i1
  }
  return %res: !stream.stream<i32>
}
