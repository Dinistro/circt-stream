// RUN: standalone-opt %s --convert-stream-to-handshake | FileCheck %s

module {
  func @filter(%in: !stream.stream<i32>) -> !stream.stream<i32> {
    %out = stream.filter(%in) : (!stream.stream<i32>) -> !stream.stream<i32> {
    ^bb0(%val: i32):
      %c0_i32 = arith.constant 0 : i32
      %0 = arith.cmpi sgt, %val, %c0_i32 : i32
      stream.yield %0 : i1
    }
    return %out : !stream.stream<i32>
  }

  // CHECK: handshake.func private @[[LABEL:.*]](%{{.*}}: i32, %{{.*}}: none, ...) -> (i32, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "outCtrl"]} {
  // CHECK-NEXT:   %{{.*}}:2 = fork [2] %{{.*}} : i32
  // CHECK-NEXT:   %{{.*}} = merge %{{.*}}#1 : i32
  // CHECK-NEXT:   %{{.*}}:2 = fork [2] %{{.*}} : none
  // CHECK-NEXT:   %{{.*}} = constant %{{.*}}#0 {value = 0 : i32} : i32
  // CHECK-NEXT:   %{{.*}} = arith.cmpi sgt, %{{.*}}, %{{.*}} : i32
  // CHECK-NEXT:   %{{.*}}, %{{.*}} = cond_br %{{.*}}, %{{.*}}#0 : i32
  // CHECK-NEXT:   sink %{{.*}} : i32
  // CHECK-NEXT:   return %{{.*}}, %{{.*}}#1 : i32, none
  // CHECK-NEXT: }
  // CHECK-NEXT: handshake.func @filter(%{{.*}}: i32, %{{.*}}: none, ...) -> (i32, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "outCtrl"]} {
  // CHECK-NEXT:  %{{.*}}:2 = instance @[[LABEL]](%{{.*}}, %{{.*}}) : (i32, none) -> (i32, none)
  // CHECK-NEXT:  return %{{.*}}#0, %{{.*}}#1 : i32, none
  // CHECK-NEXT: }
}
