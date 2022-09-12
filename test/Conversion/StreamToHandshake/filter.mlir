// RUN: stream-opt %s --convert-stream-to-handshake | FileCheck %s

module {
  func.func @filter(%in: !stream.stream<i32>) -> !stream.stream<i32> {
    %out = stream.filter(%in) : (!stream.stream<i32>) -> !stream.stream<i32> {
    ^bb0(%val: i32):
      %c0_i32 = arith.constant 0 : i32
      %0 = arith.cmpi sgt, %val, %c0_i32 : i32
      stream.yield %0 : i1
    }
    return %out : !stream.stream<i32>
  }

  // CHECK: handshake.func private @[[LABEL:.*]](%{{.*}}: tuple<i32, i1>, %{{.*}}: none, ...) -> (tuple<i32, i1>, none)
// CHECK-NEXT:   %{{.*}}:2 = fork [2] %{{.*}} : none
// CHECK-NEXT:   %{{.*}}:2 = unpack %{{.*}} : tuple<i32, i1>
// CHECK-NEXT:   %{{.*}}:2 = fork [2] %{{.*}} : i1
// CHECK-NEXT:   %{{.*}}:2 = fork [2] %{{.*}} : i32
// CHECK-NEXT:   %{{.*}} = merge %{{.*}}#0 : i32
// CHECK-NEXT:   %{{.*}} = constant %{{.*}}#0 {value = 0 : i32} : i32
// CHECK-NEXT:   %{{.*}} = arith.cmpi sgt, %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:   %{{.*}} = pack %{{.*}}#1, %{{.*}}#1 : tuple<i32, i1>
// CHECK-NEXT:   %{{.*}} = arith.ori %{{.*}}, %{{.*}}#0 : i1
// CHECK-NEXT:   %{{.*}}:2 = fork [2] %{{.*}} : i1
// CHECK-NEXT:   %{{.*}}, %{{.*}} = cond_br %{{.*}}#1, %{{.*}} : tuple<i32, i1>
// CHECK-NEXT:   sink %{{.*}} : tuple<i32, i1>
// CHECK-NEXT:   %{{.*}}, %{{.*}} = cond_br %{{.*}}#0, %{{.*}}#1 : none
// CHECK-NEXT:   sink %{{.*}} : none
// CHECK-NEXT:   return %{{.*}}, %{{.*}} : tuple<i32, i1>, none
// CHECK-NEXT: }
// CHECK-NEXT:  handshake.func @filter(%{{.*}}: tuple<i32, i1>, %{{.*}}: none, ...) -> (tuple<i32, i1>, none)
// CHECK-NEXT:    %{{.*}}:2 = instance @[[LABEL]](%{{.*}}, %{{.*}}) : (tuple<i32, i1>, none) -> (tuple<i32, i1>, none)
// CHECK-NEXT:    return %{{.*}}#0, %{{.*}}#1 : tuple<i32, i1>, none
// CHECK-NEXT:  }
}
