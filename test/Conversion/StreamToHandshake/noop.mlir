// RUN: standalone-opt %s --convert-stream-to-handshake | FileCheck %s

module {
  func.func @noop(%in: !stream.stream<i32>) -> !stream.stream<i32> {
    return %in : !stream.stream<i32>
  }
  // CHECK: handshake.func @noop(%[[VAL:.*]]: i32, %[[EOS:.*]]: i1, %[[CTRL:.*]]: none, ...) -> (i32, i1, none) attributes {argNames = ["in0", "in1", "inCtrl"], resNames = ["out0", "out1", "outCtrl"]} {
// CHECK-NEXT:    return %[[VAL]], %[[EOS]], %[[CTRL]] : i32, i1, none
// CHECK-NEXT:  }
}
