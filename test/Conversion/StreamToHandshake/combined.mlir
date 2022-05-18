// RUN: standalone-opt %s --convert-stream-to-handshake | FileCheck %s

module {
  func.func @combined(%in: !stream.stream<i512>) -> !stream.stream<i512> {
    %tmp = stream.map(%in) : (!stream.stream<i512>) -> !stream.stream<i512> {
    ^0(%val : i512):
        %0 = arith.constant 42 : i512
        %r = arith.addi %0, %val : i512
        stream.yield %r : i512
    }
    %res = stream.filter(%tmp) : (!stream.stream<i512>) -> !stream.stream<i512> {
    ^bb0(%val: i512):
      %c0_i512 = arith.constant 0 : i512
      %0 = arith.cmpi sgt, %val, %c0_i512 : i512
      stream.yield %0 : i1
    }
    return %res : !stream.stream<i512>
  }

  // CHECK:  handshake.func private @[[LABEL_FILTER:.*]](%{{.*}}: i512, %{{.*}}: none, ...) -> (i512, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "outCtrl"]} {
  // CHECK-NEXT:    %{{.*}}:2 = fork [2] %{{.*}} : i512
  // CHECK-NEXT:    %{{.*}} = merge %{{.*}}#1 : i512
  // CHECK-NEXT:    %{{.*}}:2 = fork [2] %{{.*}} : none
  // CHECK-NEXT:    %{{.*}} = constant %{{.*}}#0 {value = 0 : i512} : i512
  // CHECK-NEXT:    %{{.*}} = arith.cmpi sgt, %{{.*}}, %{{.*}} : i512
  // CHECK-NEXT:    %{{.*}}, %{{.*}} = cond_br %{{.*}}, %{{.*}}#0 : i512
  // CHECK-NEXT:    sink %{{.*}} : i512
  // CHECK-NEXT:    return %{{.*}}, %{{.*}}#1 : i512, none
  // CHECK-NEXT:  }
  // CHECK-NEXT:  handshake.func private @[[LABEL_MAP:.*]](%{{.*}}: i512, %{{.*}}: none, ...) -> (i512, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "outCtrl"]} {
  // CHECK-NEXT:    %{{.*}} = merge %{{.*}} : i512
  // CHECK-NEXT:    %{{.*}}:2 = fork [2] %{{.*}} : none
  // CHECK-NEXT:    %{{.*}} = constant %{{.*}}#0 {value = 42 : i512} : i512
  // CHECK-NEXT:    %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i512
  // CHECK-NEXT:    return %{{.*}}, %{{.*}}#1 : i512, none
  // CHECK-NEXT:  }
  // CHECK-NEXT:  handshake.func @combined(%{{.*}}: i512, %{{.*}}: none, ...) -> (i512, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "outCtrl"]} {
  // CHECK-NEXT:    %{{.*}}:2 = instance @[[LABEL_MAP]](%{{.*}}, %{{.*}}) : (i512, none) -> (i512, none)
  // CHECK-NEXT:    %{{.*}}:2 = instance @[[LABEL_FILTER]](%{{.*}}#0, %{{.*}}#1) : (i512, none) -> (i512, none)
  // CHECK-NEXT:    return %{{.*}}#0, %{{.*}}#1 : i512, none
  // CHECK-NEXT:  }
}
