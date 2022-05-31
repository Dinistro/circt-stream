// RUN: standalone-opt %s --convert-stream-to-handshake | FileCheck %s
func.func @reduce(%in: !stream.stream<i64>) -> !stream.stream<i64> {
  %res = stream.reduce(%in) {initValue = 0 : i64}: (!stream.stream<i64>) -> !stream.stream<i64> {
  ^0(%acc: i64, %val: i64):
    %r = arith.addi %acc, %val : i64
    stream.yield %r : i64
  }
  return %res : !stream.stream<i64>
}
// CHECK:       handshake.func private @[[LABEL:.*]](%{{.*}}: tuple<i64, i1>, %{{.*}}: none, ...) -> (tuple<i64, i1>, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "outCtrl"]} {
// CHECK-NEXT:    %{{.*}}:2 = unpack %{{.*}} : tuple<i64, i1>
// CHECK-NEXT:    %{{.*}}:4 = fork [4] %{{.*}}#1 : i1
// CHECK-NEXT:    %{{.*}} = merge %{{.*}} : i64
// CHECK-NEXT:    %{{.*}} = merge %{{.*}}#0 : i64
// CHECK-NEXT:    %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i64
// CHECK-NEXT:    %{{.*}} = buffer [1] seq %{{.*}} {initValues = [0]} : i64
// CHECK-NEXT:    %{{.*}}, %{{.*}} = cond_br %{{.*}}#3, %{{.*}} : i64
// CHECK-NEXT:    %{{.*}}:2 = fork [2] %{{.*}} : i64
// CHECK-NEXT:    %{{.*}}, %{{.*}} = cond_br %{{.*}}#1, %{{.*}}#2 : i1
// CHECK-NEXT:    sink %{{.*}} : i1
// CHECK-NEXT:    %{{.*}}, %{{.*}} = cond_br %{{.*}}#0, %{{.*}} : none
// CHECK-NEXT:    sink %{{.*}} : none
// CHECK-NEXT:    %{{.*}}:4 = fork [4] %{{.*}} : none
// CHECK-NEXT:    %{{.*}} = constant %{{.*}}#3 {value = false} : i1
// CHECK-NEXT:    %{{.*}} = pack %{{.*}}#1, %{{.*}} : tuple<i64, i1>
// CHECK-NEXT:    %{{.*}} = pack %{{.*}}#0, %{{.*}} : tuple<i64, i1>
// CHECK-NEXT:    %{{.*}} = constant %{{.*}}#2 {value = false} : i1
// CHECK-NEXT:    %{{.*}} = buffer [2] seq %{{.*}} {initValues = [1, 0]} : i32
// CHECK-NEXT:    %{{.*}}:2 = fork [2] %{{.*}} : i32
// CHECK-NEXT:    %{{.*}} = mux %{{.*}}#1 [%{{.*}}, %{{.*}}] : i32, tuple<i64, i1>
// CHECK-NEXT:    %{{.*}} = mux %{{.*}}#0 [%{{.*}}#0, %{{.*}}#1] : i32, none
// CHECK-NEXT:    return %{{.*}}, %{{.*}} : tuple<i64, i1>, none
// CHECK-NEXT:  }
// CHECK-NEXT:  handshake.func @reduce(%{{.*}}: tuple<i64, i1>, %{{.*}}: none, ...) -> (tuple<i64, i1>, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "outCtrl"]} {
// CHECK-NEXT:    %{{.*}}:2 = instance @[[LABEL]](%{{.*}}, %{{.*}}) : (tuple<i64, i1>, none) -> (tuple<i64, i1>, none)
// CHECK-NEXT:    return %{{.*}}#0, %{{.*}}#1 : tuple<i64, i1>, none
// CHECK-NEXT:  }
