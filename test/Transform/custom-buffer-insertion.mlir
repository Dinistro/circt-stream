// RUN: stream-opt %s --custom-buffer-insertion | FileCheck %s

// CHECK-LABEL:   handshake.func private @stream_reduce(
// CHECK-SAME:                                          %[[VAL_0:.*]]: tuple<i64, i1>,
// CHECK-SAME:                                          %[[VAL_1:.*]]: none, ...) -> (tuple<i64, i1>, none)
// CHECK:           %[[VAL_2:.*]]:2 = unpack %[[VAL_0]] : tuple<i64, i1>
// CHECK:           %[[VAL_3:.*]] = buffer [1] seq %[[VAL_2]]#1 : i1
// CHECK:           %[[VAL_4:.*]] = buffer [1] seq %[[VAL_2]]#0 : i64
// CHECK:           %[[VAL_5:.*]]:4 = fork [4] %[[VAL_3]] : i1
// CHECK:           %[[VAL_6:.*]] = buffer [1] seq %[[VAL_5]]#3 : i1
// CHECK:           %[[VAL_7:.*]] = buffer [1] seq %[[VAL_5]]#2 : i1
// CHECK:           %[[VAL_8:.*]] = buffer [1] seq %[[VAL_5]]#1 : i1
// CHECK:           %[[VAL_9:.*]] = buffer [1] seq %[[VAL_5]]#0 : i1
// CHECK:           %[[VAL_10:.*]] = buffer [1] seq %[[VAL_11:.*]] {initValues = [0]} : i64
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]] = cond_br %[[VAL_6]], %[[VAL_10]] : i64
// CHECK:           %[[VAL_14:.*]] = buffer [1] seq %[[VAL_12]] : i64
// CHECK:           %[[VAL_15:.*]]:2 = fork [2] %[[VAL_14]] : i64
// CHECK:           %[[VAL_16:.*]] = buffer [1] seq %[[VAL_15]]#1 : i64
// CHECK:           %[[VAL_17:.*]] = buffer [1] seq %[[VAL_15]]#0 : i64
// CHECK:           %[[VAL_18:.*]], %[[VAL_19:.*]] = cond_br %[[VAL_8]], %[[VAL_7]] : i1
// CHECK:           %[[VAL_20:.*]] = buffer [1] seq %[[VAL_19]] : i1
// CHECK:           %[[VAL_21:.*]] = buffer [1] seq %[[VAL_18]] : i1
// CHECK:           sink %[[VAL_20]] : i1
// CHECK:           %[[VAL_22:.*]], %[[VAL_23:.*]] = cond_br %[[VAL_9]], %[[VAL_1]] : none
// CHECK:           %[[VAL_24:.*]] = buffer [1] seq %[[VAL_23]] : none
// CHECK:           %[[VAL_25:.*]] = buffer [1] seq %[[VAL_22]] : none
// CHECK:           sink %[[VAL_24]] : none
// CHECK:           %[[VAL_26:.*]]:3 = fork [3] %[[VAL_25]] : none
// CHECK:           %[[VAL_27:.*]] = buffer [1] seq %[[VAL_26]]#2 : none
// CHECK:           %[[VAL_28:.*]] = buffer [1] seq %[[VAL_26]]#1 : none
// CHECK:           %[[VAL_29:.*]] = buffer [1] seq %[[VAL_26]]#0 : none
// CHECK:           %[[VAL_30:.*]]:2 = fork [2] %[[VAL_13]] : i64
// CHECK:           %[[VAL_31:.*]]:2 = fork [2] %[[VAL_4]] : i64
// CHECK:           %[[VAL_32:.*]] = buffer [1] seq %[[VAL_31]]#1 : i64
// CHECK:           %[[VAL_33:.*]] = buffer [1] seq %[[VAL_31]]#0 : i64
// CHECK:           %[[VAL_34:.*]] = arith.cmpi slt, %[[VAL_30]]#0, %[[VAL_33]] : i64
// CHECK:           %[[VAL_11]] = select %[[VAL_34]], %[[VAL_32]], %[[VAL_30]]#1 : i64
// CHECK:           %[[VAL_35:.*]] = constant %[[VAL_27]] {value = false} : i1
// CHECK:           %[[VAL_36:.*]] = buffer [1] seq %[[VAL_35]] : i1
// CHECK:           %[[VAL_37:.*]] = pack %[[VAL_16]], %[[VAL_36]] : tuple<i64, i1>
// CHECK:           %[[VAL_38:.*]] = buffer [1] seq %[[VAL_37]] : tuple<i64, i1>
// CHECK:           %[[VAL_39:.*]] = pack %[[VAL_17]], %[[VAL_21]] : tuple<i64, i1>
// CHECK:           %[[VAL_40:.*]] = buffer [1] seq %[[VAL_39]] : tuple<i64, i1>
// CHECK:           %[[VAL_41:.*]]:2 = fork [2] %[[VAL_40]] : tuple<i64, i1>
// CHECK:           %[[VAL_42:.*]] = buffer [1] seq %[[VAL_41]]#1 : tuple<i64, i1>
// CHECK:           %[[VAL_43:.*]] = buffer [1] seq %[[VAL_41]]#0 : tuple<i64, i1>
// CHECK:           %[[VAL_44:.*]] = constant %[[VAL_28]] {value = false} : i1
// CHECK:           %[[VAL_45:.*]] = buffer [2] seq %[[VAL_44]] {initValues = [1, 0]} : i1
// CHECK:           %[[VAL_46:.*]]:2 = fork [2] %[[VAL_45]] : i1
// CHECK:           %[[VAL_47:.*]] = buffer [1] seq %[[VAL_46]]#1 : i1
// CHECK:           %[[VAL_48:.*]] = buffer [1] seq %[[VAL_46]]#0 : i1
// CHECK:           %[[VAL_49:.*]] = mux %[[VAL_47]] {{\[}}%[[VAL_38]], %[[VAL_42]]] : i1, tuple<i64, i1>
// CHECK:           %[[VAL_50:.*]] = buffer [1] seq %[[VAL_49]] : tuple<i64, i1>
// CHECK:           %[[VAL_51:.*]] = join %[[VAL_43]] : tuple<i64, i1>
// CHECK:           %[[VAL_52:.*]] = buffer [1] seq %[[VAL_51]] : none
// CHECK:           %[[VAL_53:.*]] = mux %[[VAL_48]] {{\[}}%[[VAL_29]], %[[VAL_52]]] : i1, none
// CHECK:           %[[VAL_54:.*]] = buffer [1] seq %[[VAL_53]] : none
// CHECK:           return %[[VAL_50]], %[[VAL_54]] : tuple<i64, i1>, none
// CHECK:         }

handshake.func private @stream_reduce(%arg0: tuple<i64, i1>, %arg1: none, ...) -> (tuple<i64, i1>, none) {
  %0:2 = unpack %arg0 : tuple<i64, i1>
  %1:4 = fork [4] %0#1 : i1
  %2 = buffer [1] seq %8 {initValues = [0]} : i64
  %trueResult, %falseResult = cond_br %1#3, %2 : i64
  %3:2 = fork [2] %trueResult : i64
  %trueResult_0, %falseResult_1 = cond_br %1#1, %1#2 : i1
  sink %falseResult_1 : i1
  %trueResult_2, %falseResult_3 = cond_br %1#0, %arg1 : none
  sink %falseResult_3 : none
  %4:3 = fork [3] %trueResult_2 : none
  %5:2 = fork [2] %falseResult : i64
  %6:2 = fork [2] %0#0 : i64
  %7 = arith.cmpi slt, %5#0, %6#0 : i64
  %8 = select %7, %6#1, %5#1 : i64
  %9 = constant %4#2 {value = false} : i1
  %10 = pack %3#1, %9 : tuple<i64, i1>
  %11 = pack %3#0, %trueResult_0 : tuple<i64, i1>
  %12:2 = fork [2] %11 : tuple<i64, i1>
  %13 = constant %4#1 {value = false} : i1
  %14 = buffer [2] seq %13 {initValues = [1, 0]} : i1
  %15:2 = fork [2] %14 : i1
  %16 = mux %15#1 [%10, %12#1] : i1, tuple<i64, i1>
  %17 = join %12#0 : tuple<i64, i1>
  %18 = mux %15#0 [%4#0, %17] : i1, none
  return %16, %18 : tuple<i64, i1>, none
}
