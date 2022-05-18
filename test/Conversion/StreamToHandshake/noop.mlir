// RUN: standalone-opt %s --convert-stream-to-handshake | FileCheck %s

module {
  func.func @noop(%in: !stream.stream<i32>) -> !stream.stream<i32> {
    return %in : !stream.stream<i32>
  }
// CHECK: handshake.func @noop(%[[VAL:.*]]: i32, %[[CTRL:.*]]: none, ...) -> (i32, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "outCtrl"]} {
// CHECK-NEXT:    return %[[VAL]], %[[CTRL]] : i32, none
// CHECK-NEXT:  }
}
