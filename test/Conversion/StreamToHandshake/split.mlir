// RUN: stream-opt %s --convert-stream-to-handshake | FileCheck %s

func.func @split(%in: !stream.stream<tuple<i32, i32>>) -> (!stream.stream<i32>, !stream.stream<i32>) {
  %res0, %res1 = stream.split(%in) : (!stream.stream<tuple<i32, i32>>) -> (!stream.stream<i32>, !stream.stream<i32>) {
  ^0(%val: tuple<i32, i32>):
    %0, %1 = stream.unpack %val : tuple<i32, i32>
    stream.yield %0, %1 : i32, i32
  }
  return %res0, %res1 : !stream.stream<i32>, !stream.stream<i32>
}

// CHECK: handshake.func private @stream_split(%{{.*}}: tuple<tuple<i32, i32>, i1>, %arg1: none, ...) -> (tuple<i32, i1>, tuple<i32, i1>, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "out1", "outCtrl"]} {
// CHECK-NEXT:  %{{.*}}:2 = unpack %{{.*}} : tuple<tuple<i32, i32>, i1>
// CHECK-NEXT:  %{{.*}}:2 = fork [2] %{{.*}}#1 : i1
// CHECK-NEXT:  %{{.*}} = merge %{{.*}}#0 : tuple<i32, i32>
// CHECK-NEXT:  %{{.*}}:2 = unpack %{{.*}} : tuple<i32, i32>
// CHECK-NEXT:  %{{.*}} = pack %{{.*}}#0, %{{.*}}#1 : tuple<i32, i1>
// CHECK-NEXT:  %{{.*}} = pack %{{.*}}#1, %{{.*}}#0 : tuple<i32, i1>
// CHECK-NEXT:  return %{{.*}}, %{{.*}}, %{{.*}} : tuple<i32, i1>, tuple<i32, i1>, none
// CHECK-NEXT:}
// CHECK-NEXT:handshake.func @split(%{{.*}}: tuple<tuple<i32, i32>, i1>, %{{.*}}: none, ...) -> (tuple<i32, i1>, tuple<i32, i1>, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "out1", "outCtrl"]} {
// CHECK-NEXT:  %{{.*}}:3 = instance @stream_split(%{{.*}}, %{{.*}}) : (tuple<tuple<i32, i32>, i1>, none) -> (tuple<i32, i1>, tuple<i32, i1>, none)
// CHECK-NEXT:  return %{{.*}}#0, %{{.*}}#1, %{{.*}}#2 : tuple<i32, i1>, tuple<i32, i1>, none
// CHECK-NEXT:}
