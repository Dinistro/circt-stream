// RUN: stream-opt %s --convert-stream-to-handshake | FileCheck %s

func.func @create() -> !stream.stream<i32> {
  %out = stream.create !stream.stream<i32> [1,2,3]
  return %out : !stream.stream<i32>
}
// CHECK:  handshake.func private @[[LABEL:.*]](%{{.*}}: none, ...) -> (tuple<i32, i1>, none) attributes {argNames = ["inCtrl"], resNames = ["out0", "outCtrl"]} {
// CHECK-NEXT:    %{{.*}}:2 = fork [2] %{{.*}} : none
// CHECK-NEXT:    %{{.*}} = constant %{{.*}}#1 {value = false} : i1
// CHECK-NEXT:    %{{.*}} = buffer [1] seq %{{.*}} {initValues = [1]} : i1
// CHECK-NEXT:    %{{.*}}, %{{.*}} = cond_br %{{.*}}, %{{.*}}#0 : none
// CHECK-NEXT:    sink %{{.*}} : none
// CHECK-NEXT:    %{{.*}} = buffer [2] seq %{{.*}}#0 : none
// CHECK-NEXT:    %{{.*}} = merge %{{.*}}, %{{.*}} : none
// CHECK-NEXT:    %{{.*}}:5 = fork [5] %{{.*}} : none
// CHECK-NEXT:    %{{.*}} = constant %{{.*}}#4 {value = 0 : i32} : i32
// CHECK-NEXT:    %{{.*}} = buffer [3] seq %{{.*}} {initValues = [3, 2, 1]} : i32
// CHECK-NEXT:    %{{.*}} = buffer [1] seq %{{.*}} {initValues = [0]} : i64
// CHECK-NEXT:    %{{.*}}:2 = fork [2] %{{.*}} : i64
// CHECK-NEXT:    %{{.*}} = constant %{{.*}}#3 {value = 1 : i64} : i64
// CHECK-NEXT:    %{{.*}} = constant %{{.*}}#2 {value = 3 : i64} : i64
// CHECK-NEXT:    %{{.*}} = arith.cmpi eq, %{{.*}}#1, %{{.*}} : i64
// CHECK-NEXT:    %{{.*}} = arith.addi %{{.*}}#0, %{{.*}} : i64
// CHECK-NEXT:    %{{.*}} = pack %{{.*}}, %{{.*}} : tuple<i32, i1>
// CHECK-NEXT:    return %{{.*}}, %{{.*}}#1 : tuple<i32, i1>, none
// CHECK-NEXT:  }
