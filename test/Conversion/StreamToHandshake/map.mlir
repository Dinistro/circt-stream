// RUN: standalone-opt %s --convert-stream-to-handshake | FileCheck %s

module {
    func @map(%in: !stream.stream<i32>) -> !stream.stream<i32> {
        %res = stream.map(%in) : (!stream.stream<i32>) -> !stream.stream<i32> {
        ^0(%val : i32):
            %0 = arith.constant 1 : i32
            %r = arith.addi %0, %val : i32
            stream.yield %r : i32
        }
        return %res : !stream.stream<i32>
    }

// CHECK:  handshake.func private @[[LABEL:.*]](%{{.*}}: i32, %{{.*}}: none, ...) -> (i32, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "outCtrl"]} {
// CHECK-NEXT:    %{{.*}} = merge %{{.*}} : i32
// CHECK-NEXT:    %{{.*}}:2 = fork [2] %{{.*}} : none
// CHECK-NEXT:    %{{.*}} = constant %{{.*}}#0 {value = 1 : i32} : i32
// CHECK-NEXT:    %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:    return %{{.*}}, %{{.*}}#1 : i32, none
// CHECK-NEXT:  }
// CHECK-NEXT:  handshake.func @map(%arg0: i32, %arg1: none, ...) -> (i32, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "outCtrl"]} {
// CHECK-NEXT:    %{{.*}}:2 = instance @[[LABEL]](%arg{{.*}}, %{{.*}}) : (i32, none) -> (i32, none)
// CHECK-NEXT:    return %{{.*}}#0, %{{.*}}#1 : i32, none
// CHECK-NEXT:  }
}
