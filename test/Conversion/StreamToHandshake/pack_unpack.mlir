// RUN: standalone-opt %s --convert-stream-to-handshake | FileCheck %s

func.func @map(%in: !stream.stream<tuple<i32, i64>>) -> !stream.stream<tuple<i64, i32>> {
  %res = stream.map(%in) : (!stream.stream<tuple<i32, i64>>) -> !stream.stream<tuple<i64, i32>> {
  ^0(%val : tuple<i32, i64>):
    %a, %b = stream.unpack %val : tuple<i32, i64>
    %r = stream.pack %b, %a : tuple<i64, i32>
    stream.yield %r : tuple<i64, i32>
  }
  return %res : !stream.stream<tuple<i64, i32>>
}

// CHECK: handshake.func private @[[LABEL:.*]](%{{.*}}: tuple<i32, i64>, %{{.*}}: none, ...) -> (tuple<i64, i32>, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "outCtrl"]} {
// CHECK-NEXT:   %{{.*}} = merge %{{.*}} : tuple<i32, i64>
// CHECK-NEXT:   %{{.*}}:2 = unpack %{{.*}} : tuple<i32, i64>
// CHECK-NEXT:   %{{.*}} = pack %{{.*}}#1, %{{.*}}#0 : tuple<i64, i32>
// CHECK-NEXT:   return %{{.*}}, %{{.*}} : tuple<i64, i32>, none
// CHECK-NEXT: }
// CHECK-NEXT: handshake.func @map(%{{.*}}: tuple<i32, i64>, %{{.*}}: none, ...) -> (tuple<i64, i32>, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "outCtrl"]} {
// CHECK-NEXT:   %{{.*}}:2 = instance @[[LABEL]](%{{.*}}, %{{.*}}) : (tuple<i32, i64>, none) -> (tuple<i64, i32>, none)
// CHECK-NEXT:   return %{{.*}}#0, %{{.*}}#1 : tuple<i64, i32>, none
// CHECK-NEXT: }
