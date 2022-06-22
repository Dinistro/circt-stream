// REQUIRES: verilator
// RUN: stream-opt %s --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --verilog > %t.sv && \
// RUN: circt-rtl-sim.py %t.sv %S/driver_out_i64.sv %S/driver.cpp --no-default-driver --top driver | FileCheck %s
// CHECK: Element={{.*}}1
// CHECK-NEXT: EOS
module {
  handshake.func @top(%arg0: none, ...) -> (tuple<i64, i1>, none, none) attributes {argNames = ["inCtrl"], resNames = ["out0", "out1", "outCtrl"]} {

    %0:3 = fork [3] %arg0 : none
    sink %0#1 : none


    //%1 = constant %0#1 {value = false} : i1
    //%2 = buffer [1] seq %1 {initValues = [1]} : i1
    //%trueResult, %falseResult = cond_br %2, %0#0 : none


    //sink %falseResult : none
    %3 = buffer [2] seq %6#0 : none
    //%4 = merge %trueResult, %3 : none
    %4 = merge %0#0, %3 : none
    %5 = buffer [2] fifo %4 : none
    %6:5 = fork [5] %5 : none




    %7 = constant %6#4 {value = 0 : i64} : i64
    %8 = buffer [1] seq %7 {initValues = [1]} : i64
    %9 = buffer [1] seq %14 {initValues = [0]} : i64
    %10:2 = fork [2] %9 : i64


    %11 = constant %6#3 {value = 1 : i64} : i64

    %12 = constant %6#2 {value = 1 : i64} : i64

    %13 = arith.cmpi eq, %10#1, %12 : i64

    %14 = arith.addi %10#0, %11 : i64
    %15 = pack %8, %13 : tuple<i64, i1>

    return %15, %6#1, %0#2 : tuple<i64, i1>, none, none
    //%out0 = buffer [2] fifo %15 : tuple<i64, i1>
    //%out1 = buffer [2] fifo %6#1 : none
    //%out2 = buffer [2] fifo %0#2 : none

    //return %out0, %out1, %out2 : tuple<i64, i1>, none, none

    //%out0 = buffer [2] fifo %15 : tuple<i64, i1>
    //return %out0, %6#1, %0#2 : tuple<i64, i1>, none, none
  }
}

