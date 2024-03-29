// RUN: stream-opt %s --convert-stream-to-handshake | FileCheck %s

module {
  func.func @map(%in: !stream.stream<i32>) -> !stream.stream<i32> {
    %tmp = stream.map(%in) : (!stream.stream<i32>) -> !stream.stream<i32> {
    ^0(%val : i32):
      %0 = arith.constant 1 : i32
      %r = arith.addi %0, %val : i32
      stream.yield %r : i32
    }
    %res = stream.map(%tmp) : (!stream.stream<i32>) -> !stream.stream<i32> {
    ^0(%val : i32):
      %0 = arith.constant 10 : i32
      %r = arith.muli %0, %val : i32
      stream.yield %r : i32
    }
    return %res : !stream.stream<i32>
  }

  // CHECK:  handshake.func private @[[LABEL_1:.*]](%{{.*}}: tuple<i32, i1>, ...) -> tuple<i32, i1>

  // CHECK:  handshake.func private @[[LABEL_0:.*]](%{{.*}}: tuple<i32, i1>, ...) -> tuple<i32, i1>

  // CHECK:  handshake.func @map(%{{.*}}: tuple<i32, i1>, ...) -> tuple<i32, i1>
// CHECK-NEXT:    %{{.*}} = instance @[[LABEL_0]](%{{.*}}) : (tuple<i32, i1>) -> tuple<i32, i1>
// CHECK-NEXT:    %{{.*}} = instance @[[LABEL_1]](%{{.*}}) : (tuple<i32, i1>) -> tuple<i32, i1>
// CHECK-NEXT:    return %{{.*}} : tuple<i32, i1>
// CHECK-NEXT:  }
}
