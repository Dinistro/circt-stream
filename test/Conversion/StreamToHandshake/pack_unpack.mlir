// RUN: stream-opt %s --convert-stream-to-handshake | FileCheck %s

func.func @map(%in: !stream.stream<tuple<i32, i64>>) -> !stream.stream<tuple<i64, i32>> {
  %res = stream.map(%in) : (!stream.stream<tuple<i32, i64>>) -> !stream.stream<tuple<i64, i32>> {
  ^0(%val : tuple<i32, i64>):
    %a, %b = stream.unpack %val : tuple<i32, i64>
    %r = stream.pack %b, %a : tuple<i64, i32>
    stream.yield %r : tuple<i64, i32>
  }
  return %res : !stream.stream<tuple<i64, i32>>
}

// CHECK: handshake.func private @[[LABEL:.*]](%{{.*}}: tuple<tuple<i32, i64>, i1>, %{{.*}}: none, ...) -> (tuple<tuple<i64, i32>, i1>, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "outCtrl"]} {
// CHECK-NEXT:   %{{.*}}:2 = unpack %{{.*}} : tuple<tuple<i32, i64>, i1>
// CHECK-NEXT:   %{{.*}} = merge %{{.*}} : tuple<i32, i64>
// CHECK-NEXT:   %{{.*}}:2 = unpack %{{.*}} : tuple<i32, i64>
// CHECK-NEXT:   %{{.*}} = pack %{{.*}}#1, %{{.*}}#0 : tuple<i64, i32>
// CHECK-NEXT:   %{{.*}} = pack %{{.*}}, %{{.*}}#1 : tuple<tuple<i64, i32>, i1>
// CHECK-NEXT:   return %{{.*}}, %{{.*}} : tuple<tuple<i64, i32>, i1>, none
// CHECK-NEXT: }
// CHECK-NEXT: handshake.func @map(%{{.*}}: tuple<tuple<i32, i64>, i1>, %{{.*}}: none, ...) -> (tuple<tuple<i64, i32>, i1>, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "outCtrl"]} {
// CHECK-NEXT:   %{{.*}}:2 = instance @[[LABEL]](%{{.*}}, %{{.*}}) : (tuple<tuple<i32, i64>, i1>, none) -> (tuple<tuple<i64, i32>, i1>, none)
// CHECK-NEXT:   return %{{.*}}#0, %{{.*}}#1 : tuple<tuple<i64, i32>, i1>, none
// CHECK-NEXT: }
