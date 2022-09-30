// RUN: stream-opt %s --custom-buffer-insertion | FileCheck %s

// CHECK-LABEL:   handshake.func private @stream_reduce(
// CHECK-SAME:                                          %[[VAL_0:.*]]: tuple<i64, i1>,
// CHECK-SAME:                                          %[[VAL_1:.*]]: none, ...) -> (tuple<i64, i1>, none)
// CHECK:           %[[VAL_2:.*]] = buffer [10] fifo %[[VAL_1]] : none
// CHECK:           %[[VAL_3:.*]] = buffer [1] seq %[[VAL_2]] : none
// CHECK:           %[[VAL_4:.*]] = buffer [10] fifo %[[VAL_0]] : tuple<i64, i1>
// CHECK:           %[[VAL_5:.*]] = buffer [1] seq %[[VAL_4]] : tuple<i64, i1>
// CHECK:           %[[VAL_6:.*]]:2 = unpack %[[VAL_5]] : tuple<i64, i1>
// CHECK:           %[[VAL_7:.*]] = buffer [10] fifo %[[VAL_6]]#1 : i1
// CHECK:           %[[VAL_8:.*]] = buffer [1] seq %[[VAL_7]] : i1
// CHECK:           %[[VAL_9:.*]] = buffer [10] fifo %[[VAL_6]]#0 : i64
// CHECK:           %[[VAL_10:.*]] = buffer [1] seq %[[VAL_9]] : i64
// CHECK:           %[[VAL_11:.*]]:4 = fork [4] %[[VAL_8]] : i1
// CHECK:           %[[VAL_12:.*]] = buffer [10] fifo %[[VAL_11]]#3 : i1
// CHECK:           %[[VAL_13:.*]] = buffer [1] seq %[[VAL_12]] : i1
// CHECK:           %[[VAL_14:.*]] = buffer [10] fifo %[[VAL_11]]#2 : i1
// CHECK:           %[[VAL_15:.*]] = buffer [1] seq %[[VAL_14]] : i1
// CHECK:           %[[VAL_16:.*]] = buffer [10] fifo %[[VAL_11]]#1 : i1
// CHECK:           %[[VAL_17:.*]] = buffer [1] seq %[[VAL_16]] : i1
// CHECK:           %[[VAL_18:.*]] = buffer [10] fifo %[[VAL_11]]#0 : i1
// CHECK:           %[[VAL_19:.*]] = buffer [1] seq %[[VAL_18]] : i1
// CHECK:           %[[VAL_20:.*]] = buffer [1] seq %[[VAL_21:.*]] {initValues = [0]} : i64
// CHECK:           %[[VAL_22:.*]], %[[VAL_23:.*]] = cond_br %[[VAL_13]], %[[VAL_20]] : i64
// CHECK:           %[[VAL_24:.*]] = buffer [10] fifo %[[VAL_22]] : i64
// CHECK:           %[[VAL_25:.*]] = buffer [1] seq %[[VAL_24]] : i64
// CHECK:           %[[VAL_26:.*]]:2 = fork [2] %[[VAL_25]] : i64
// CHECK:           %[[VAL_27:.*]] = buffer [10] fifo %[[VAL_26]]#1 : i64
// CHECK:           %[[VAL_28:.*]] = buffer [1] seq %[[VAL_27]] : i64
// CHECK:           %[[VAL_29:.*]] = buffer [10] fifo %[[VAL_26]]#0 : i64
// CHECK:           %[[VAL_30:.*]] = buffer [1] seq %[[VAL_29]] : i64
// CHECK:           %[[VAL_31:.*]], %[[VAL_32:.*]] = cond_br %[[VAL_17]], %[[VAL_15]] : i1
// CHECK:           %[[VAL_33:.*]] = buffer [10] fifo %[[VAL_32]] : i1
// CHECK:           %[[VAL_34:.*]] = buffer [1] seq %[[VAL_33]] : i1
// CHECK:           %[[VAL_35:.*]] = buffer [10] fifo %[[VAL_31]] : i1
// CHECK:           %[[VAL_36:.*]] = buffer [1] seq %[[VAL_35]] : i1
// CHECK:           sink %[[VAL_34]] : i1
// CHECK:           %[[VAL_37:.*]], %[[VAL_38:.*]] = cond_br %[[VAL_19]], %[[VAL_3]] : none
// CHECK:           %[[VAL_39:.*]] = buffer [10] fifo %[[VAL_38]] : none
// CHECK:           %[[VAL_40:.*]] = buffer [1] seq %[[VAL_39]] : none
// CHECK:           %[[VAL_41:.*]] = buffer [10] fifo %[[VAL_37]] : none
// CHECK:           %[[VAL_42:.*]] = buffer [1] seq %[[VAL_41]] : none
// CHECK:           sink %[[VAL_40]] : none
// CHECK:           %[[VAL_43:.*]]:3 = fork [3] %[[VAL_42]] : none
// CHECK:           %[[VAL_44:.*]] = buffer [10] fifo %[[VAL_43]]#2 : none
// CHECK:           %[[VAL_45:.*]] = buffer [1] seq %[[VAL_44]] : none
// CHECK:           %[[VAL_46:.*]] = buffer [10] fifo %[[VAL_43]]#1 : none
// CHECK:           %[[VAL_47:.*]] = buffer [1] seq %[[VAL_46]] : none
// CHECK:           %[[VAL_48:.*]] = buffer [10] fifo %[[VAL_43]]#0 : none
// CHECK:           %[[VAL_49:.*]] = buffer [1] seq %[[VAL_48]] : none
// CHECK:           %[[VAL_50:.*]]:2 = fork [2] %[[VAL_23]] : i64
// CHECK:           %[[VAL_51:.*]]:2 = fork [2] %[[VAL_10]] : i64
// CHECK:           %[[VAL_52:.*]] = buffer [10] fifo %[[VAL_51]]#1 : i64
// CHECK:           %[[VAL_53:.*]] = buffer [1] seq %[[VAL_52]] : i64
// CHECK:           %[[VAL_54:.*]] = buffer [10] fifo %[[VAL_51]]#0 : i64
// CHECK:           %[[VAL_55:.*]] = buffer [1] seq %[[VAL_54]] : i64
// CHECK:           %[[VAL_56:.*]] = arith.cmpi slt, %[[VAL_50]]#0, %[[VAL_55]] : i64
// CHECK:           %[[VAL_21]] = select %[[VAL_56]], %[[VAL_53]], %[[VAL_50]]#1 : i64
// CHECK:           %[[VAL_57:.*]] = constant %[[VAL_45]] {value = false} : i1
// CHECK:           %[[VAL_58:.*]] = buffer [10] fifo %[[VAL_57]] : i1
// CHECK:           %[[VAL_59:.*]] = buffer [1] seq %[[VAL_58]] : i1
// CHECK:           %[[VAL_60:.*]] = pack %[[VAL_28]], %[[VAL_59]] : tuple<i64, i1>
// CHECK:           %[[VAL_61:.*]] = buffer [10] fifo %[[VAL_60]] : tuple<i64, i1>
// CHECK:           %[[VAL_62:.*]] = buffer [1] seq %[[VAL_61]] : tuple<i64, i1>
// CHECK:           %[[VAL_63:.*]] = pack %[[VAL_30]], %[[VAL_36]] : tuple<i64, i1>
// CHECK:           %[[VAL_64:.*]] = buffer [10] fifo %[[VAL_63]] : tuple<i64, i1>
// CHECK:           %[[VAL_65:.*]] = buffer [1] seq %[[VAL_64]] : tuple<i64, i1>
// CHECK:           %[[VAL_66:.*]]:2 = fork [2] %[[VAL_65]] : tuple<i64, i1>
// CHECK:           %[[VAL_67:.*]] = buffer [10] fifo %[[VAL_66]]#1 : tuple<i64, i1>
// CHECK:           %[[VAL_68:.*]] = buffer [1] seq %[[VAL_67]] : tuple<i64, i1>
// CHECK:           %[[VAL_69:.*]] = buffer [10] fifo %[[VAL_66]]#0 : tuple<i64, i1>
// CHECK:           %[[VAL_70:.*]] = buffer [1] seq %[[VAL_69]] : tuple<i64, i1>
// CHECK:           %[[VAL_71:.*]] = constant %[[VAL_47]] {value = false} : i1
// CHECK:           %[[VAL_72:.*]] = buffer [2] seq %[[VAL_71]] {initValues = [1, 0]} : i1
// CHECK:           %[[VAL_73:.*]]:2 = fork [2] %[[VAL_72]] : i1
// CHECK:           %[[VAL_74:.*]] = buffer [10] fifo %[[VAL_73]]#1 : i1
// CHECK:           %[[VAL_75:.*]] = buffer [1] seq %[[VAL_74]] : i1
// CHECK:           %[[VAL_76:.*]] = buffer [10] fifo %[[VAL_73]]#0 : i1
// CHECK:           %[[VAL_77:.*]] = buffer [1] seq %[[VAL_76]] : i1
// CHECK:           %[[VAL_78:.*]] = mux %[[VAL_75]] {{\[}}%[[VAL_62]], %[[VAL_68]]] : i1, tuple<i64, i1>
// CHECK:           %[[VAL_79:.*]] = buffer [10] fifo %[[VAL_78]] : tuple<i64, i1>
// CHECK:           %[[VAL_80:.*]] = buffer [1] seq %[[VAL_79]] : tuple<i64, i1>
// CHECK:           %[[VAL_81:.*]] = join %[[VAL_70]] : tuple<i64, i1>
// CHECK:           %[[VAL_82:.*]] = buffer [10] fifo %[[VAL_81]] : none
// CHECK:           %[[VAL_83:.*]] = buffer [1] seq %[[VAL_82]] : none
// CHECK:           %[[VAL_84:.*]] = mux %[[VAL_77]] {{\[}}%[[VAL_49]], %[[VAL_83]]] : i1, none
// CHECK:           %[[VAL_85:.*]] = buffer [10] fifo %[[VAL_84]] : none
// CHECK:           %[[VAL_86:.*]] = buffer [1] seq %[[VAL_85]] : none
// CHECK:           return %[[VAL_80]], %[[VAL_86]] : tuple<i64, i1>, none
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
