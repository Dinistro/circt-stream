// RUN: stream-opt %s --convert-stream-to-handshake | FileCheck %s

module {
  func.func @noop(%in: !stream.stream<i32>) -> !stream.stream<i32> {
    return %in : !stream.stream<i32>
  }
  // CHECK: handshake.func @noop(%[[VAL:.*]]: tuple<i32, i1>, %[[CTRL:.*]]: none, ...) -> (tuple<i32, i1>, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "outCtrl"]} {
// CHECK-NEXT:    return %[[VAL]], %[[CTRL]] : tuple<i32, i1>, none
// CHECK-NEXT:  }
}
