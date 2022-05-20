// RUN: standalone-opt %s --convert-stream-to-handshake | FileCheck %s
func.func @reduce(%in: !stream.stream<i64>) -> !stream.stream<i64> {
  %res = stream.reduce(%in) {initValue = 0 : i64}: (!stream.stream<i64>) -> !stream.stream<i64> {
  ^0(%acc: i64, %val: i64):
    %r = arith.addi %acc, %val : i64
    stream.yield %r : i64
  }
  return %res : !stream.stream<i64>
}
// CHECK: handshake.func private @[[LABEL:.*]](%{{.*}}: i64, %{{.*}}: i1, %{{.*}}: none, ...) -> (i64, i1, none) attributes {argNames = ["in0", "in1", "inCtrl"], resNames = ["out0", "out1", "outCtrl"]} {
// CHECK-NEXT:   %{{.*}} = buffer [1] seq %{{.*}}#1 {initValues = [0]} : i64
// CHECK-NEXT:   %{{.*}} = merge %{{.*}} : i64
// CHECK-NEXT:   %{{.*}} = merge %{{.*}} : i64
// CHECK-NEXT:   %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i64
// CHECK-NEXT:   %{{.*}}:2 = fork [2] %{{.*}} : i64
// CHECK-NEXT:   return %{{.*}}#0, %{{.*}}, %{{.*}} : i64, i1, none
// CHECK-NEXT: }
// CHECK-NEXT: handshake.func @reduce(%{{.*}}: i64, %{{.*}}: i1, %{{.*}}: none, ...) -> (i64, i1, none) attributes {argNames = ["in0", "in1", "inCtrl"], resNames = ["out0", "out1", "outCtrl"]} {
// CHECK-NEXT:   %{{.*}}:3 = instance @[[LABEL]](%{{.*}}, %{{.*}}, %{{.*}}) : (i64, i1, none) -> (i64, i1, none)
// CHECK-NEXT:   return %{{.*}}#0, %{{.*}}#1, %{{.*}}#2 : i64, i1, none
// CHECK-NEXT: }
