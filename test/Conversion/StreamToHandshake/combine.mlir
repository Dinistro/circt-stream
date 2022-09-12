// RUN: stream-opt %s --convert-stream-to-handshake | FileCheck %s

module {
  func.func @combine(%in0: !stream.stream<i32>, %in1: !stream.stream<i32>) -> (!stream.stream<i32>) {
    %res = stream.combine(%in0, %in1) : (!stream.stream<i32>, !stream.stream<i32>) -> (!stream.stream<i32>) {
    ^0(%val0: i32, %val1: i32):
    %0 = arith.addi %val0, %val1 : i32
      stream.yield %0 : i32
    }
    return %res : !stream.stream<i32>
  }

  // CHECK:  handshake.func private @[[LABEL:.*]](%{{.*}}: tuple<i32, i1>, %{{.*}}: none, %{{.*}}: tuple<i32, i1>, %{{.*}}: none, ...) -> (tuple<i32, i1>, none)
  // CHECK-NEXT:   %{{.*}}:2 = unpack %{{.*}} : tuple<i32, i1>
  // CHECK-NEXT:   %{{.*}}:2 = unpack %{{.*}} : tuple<i32, i1>
  // CHECK-NEXT:   %{{.*}} = join %{{.*}}, %{{.*}} : none
  // CHECK-NEXT:   %{{.*}} = merge %{{.*}}#0 : i32
  // CHECK-NEXT:   %{{.*}} = merge %{{.*}}#0 : i32
  // CHECK-NEXT:   %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
  // CHECK-NEXT:   %{{.*}} = arith.ori %{{.*}}#1, %{{.*}}#1 : i1
  // CHECK-NEXT:   %{{.*}} = pack %{{.*}}, %{{.*}} : tuple<i32, i1>
  // CHECK-NEXT:   return %{{.*}}, %{{.*}} : tuple<i32, i1>, none
  // CHECK-NEXT: }
  // CHECK-NEXT: handshake.func @combine(%{{.*}}: tuple<i32, i1>, %{{.*}}: none, %{{.*}}: tuple<i32, i1>, %{{.*}}: none, ...) -> (tuple<i32, i1>, none)
  // CHECK-NEXT:   %{{.*}}:2 = instance @[[LABEL]](%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (tuple<i32, i1>, none, tuple<i32, i1>, none) -> (tuple<i32, i1>, none)
  // CHECK-NEXT:   return %{{.*}}#0, %{{.*}}#1 : tuple<i32, i1>, none
  // CHECK-NEXT: }
}
