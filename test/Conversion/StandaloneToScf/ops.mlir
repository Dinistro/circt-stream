// RUN: stream-opt --convert-standalone-to-scf %s | FileCheck %s

module {
  func.func @simple(%0: i32) -> (i32) {
    %res = standalone.neg_to_zero %0 : i32
    return %res : i32
  }
  // CHECK-LABEL: func.func @simple(%{{.*}}: i32) -> i32 {
  // CHECK-NEXT: %{{.*}} = arith.constant 0 : i32
  // CHECK-NEXT: %{{.*}} = arith.cmpi sgt, %{{.*}}, %{{.*}} : i32
  // CHECK-NEXT: %{{.*}} = scf.if %0 -> (i32) {
  // CHECK-NEXT:   scf.yield %{{.*}} : i32
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   scf.yield %{{.*}} : i32
  // CHECK-NEXT: }
  // CHECK-NEXT: return %{{.*}} : i32
}
