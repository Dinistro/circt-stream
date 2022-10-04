// RUN: stream-opt %s --convert-stream-to-handshake --split-input-file | FileCheck %s

// CHECK-LABEL:   handshake.func private @stream_map(
// CHECK-SAME:                                       %[[VAL_0:.*]]: tuple<i32, i1>, ...) -> tuple<i32, i1>
// CHECK:           %[[VAL_1:.*]] = source
// CHECK:           %[[VAL_2:.*]] = constant %[[VAL_1]] {value = 0 : i32} : i32
// CHECK:           %[[VAL_3:.*]] = mux %[[VAL_4:.*]]#1 {{\[}}%[[VAL_5:.*]], %[[VAL_2]]] : i1, i32
// CHECK:           %[[VAL_6:.*]]:2 = unpack %[[VAL_0]] : tuple<i32, i1>
// CHECK:           %[[VAL_4]]:5 = fork [5] %[[VAL_6]]#1 : i1
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = cond_br %[[VAL_4]]#4, %[[VAL_6]]#0 : i32
// CHECK:           sink %[[VAL_7]] : i32
// CHECK:           %[[VAL_9:.*]]:2 = fork [2] %[[VAL_8]] : i32
// CHECK:           %[[VAL_10:.*]] = source
// CHECK:           %[[VAL_11:.*]] = constant %[[VAL_10]] {value = 0 : i32} : i32
// CHECK:           %[[VAL_12:.*]] = mux %[[VAL_4]]#3 {{\[}}%[[VAL_13:.*]], %[[VAL_11]]] : i1, i32
// CHECK:           %[[VAL_14:.*]] = buffer [1] seq %[[VAL_12]] {initValues = [0]} : i32
// CHECK:           %[[VAL_15:.*]], %[[VAL_16:.*]] = cond_br %[[VAL_4]]#2, %[[VAL_14]] : i32
// CHECK:           sink %[[VAL_15]] : i32
// CHECK:           %[[VAL_17:.*]] = join %[[VAL_9]]#1 : i32
// CHECK:           sink %[[VAL_17]] : none
// CHECK:           %[[VAL_5]] = merge %[[VAL_9]]#0 : i32
// CHECK:           %[[VAL_13]] = merge %[[VAL_16]] : i32
// CHECK:           %[[VAL_18:.*]] = pack %[[VAL_3]], %[[VAL_4]]#0 : tuple<i32, i1>
// CHECK:           return %[[VAL_18]] : tuple<i32, i1>
// CHECK:         }

// CHECK-LABEL:   handshake.func @single_reg(
// CHECK-SAME:                               %[[VAL_0:.*]]: tuple<i32, i1>, ...) -> tuple<i32, i1>
// CHECK:           %[[VAL_1:.*]] = instance @stream_map(%[[VAL_0]]) : (tuple<i32, i1>) -> tuple<i32, i1>
// CHECK:           return %[[VAL_1]] : tuple<i32, i1>
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
// CHECK-SAME:                                          %[[VAL_0:.*]]: tuple<i32, i1>, ...) -> tuple<i32, i1>
// CHECK:           %[[VAL_1:.*]]:2 = fork [2] %[[VAL_0]] : tuple<i32, i1>
// CHECK:           %[[VAL_2:.*]]:2 = unpack %[[VAL_1]]#1 : tuple<i32, i1>
// CHECK:           %[[VAL_3:.*]]:8 = fork [8] %[[VAL_2]]#1 : i1
// CHECK:           %[[VAL_4:.*]], %[[VAL_5:.*]] = cond_br %[[VAL_3]]#7, %[[VAL_2]]#0 : i32
// CHECK:           sink %[[VAL_4]] : i32
// CHECK:           %[[VAL_6:.*]]:2 = fork [2] %[[VAL_5]] : i32
// CHECK:           %[[VAL_7:.*]] = source
// CHECK:           %[[VAL_8:.*]] = constant %[[VAL_7]] {value = false} : i1
// CHECK:           %[[VAL_9:.*]] = mux %[[VAL_3]]#6 {{\[}}%[[VAL_10:.*]], %[[VAL_8]]] : i1, i1
// CHECK:           %[[VAL_11:.*]] = buffer [1] seq %[[VAL_9]] {initValues = [0]} : i1
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]] = cond_br %[[VAL_3]]#5, %[[VAL_11]] : i1
// CHECK:           sink %[[VAL_12]] : i1
// CHECK:           %[[VAL_14:.*]] = source
// CHECK:           %[[VAL_15:.*]] = constant %[[VAL_14]] {value = true} : i1
// CHECK:           %[[VAL_16:.*]] = mux %[[VAL_3]]#4 {{\[}}%[[VAL_17:.*]]#1, %[[VAL_15]]] : i1, i1
// CHECK:           %[[VAL_18:.*]] = buffer [1] seq %[[VAL_16]] {initValues = [-1]} : i1
// CHECK:           %[[VAL_19:.*]], %[[VAL_20:.*]] = cond_br %[[VAL_3]]#3, %[[VAL_18]] : i1
// CHECK:           sink %[[VAL_19]] : i1
// CHECK:           %[[VAL_21:.*]] = join %[[VAL_6]]#1 : i32
// CHECK:           sink %[[VAL_21]] : none
// CHECK:           %[[VAL_22:.*]] = merge %[[VAL_6]]#0 : i32
// CHECK:           sink %[[VAL_22]] : i32
// CHECK:           %[[VAL_23:.*]] = merge %[[VAL_13]] : i1
// CHECK:           %[[VAL_17]]:2 = fork [2] %[[VAL_23]] : i1
// CHECK:           %[[VAL_10]] = merge %[[VAL_20]] : i1
// CHECK:           %[[VAL_24:.*]], %[[VAL_25:.*]] = cond_br %[[VAL_3]]#1, %[[VAL_3]]#2 : i1
// CHECK:           sink %[[VAL_25]] : i1
// CHECK:           %[[VAL_26:.*]] = mux %[[VAL_3]]#0 {{\[}}%[[VAL_17]]#0, %[[VAL_24]]] : i1, i1
// CHECK:           %[[VAL_27:.*]], %[[VAL_28:.*]] = cond_br %[[VAL_26]], %[[VAL_1]]#0 : tuple<i32, i1>
// CHECK:           sink %[[VAL_28]] : tuple<i32, i1>
// CHECK:           return %[[VAL_27]] : tuple<i32, i1>
// CHECK:         }

// CHECK-LABEL:   handshake.func @multiple_regs(
// CHECK-SAME:                                  %[[VAL_0:.*]]: tuple<i32, i1>, ...) -> tuple<i32, i1>
// CHECK:           %[[VAL_1:.*]] = instance @stream_filter(%[[VAL_0]]) : (tuple<i32, i1>) -> tuple<i32, i1>
// CHECK:           return %[[VAL_1]] : tuple<i32, i1>
// CHECK:         }


func.func @multiple_regs(%in: !stream.stream<i32>) -> !stream.stream<i32> {
  %res = stream.filter(%in) {registers = [0 : i1, 1 : i1]}: (!stream.stream<i32>) -> !stream.stream<i32> {
  ^0(%val : i32, %reg0: i1, %reg1 : i1):
    stream.yield %reg0, %reg1, %reg0 : i1, i1, i1
  }
  return %res: !stream.stream<i32>
}
