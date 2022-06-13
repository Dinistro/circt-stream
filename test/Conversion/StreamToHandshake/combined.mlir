// RUN: stream-opt %s --convert-stream-to-handshake | FileCheck %s

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

// CHECK: handshake.func private @[[LABEL_FILTER:.*]](%{{.*}}: tuple<i512, i1>, %{{.*}}: none, %{{.*}}: none, ...) -> (tuple<i512, i1>, none, none)
// CHECK-NEXT:    %{{.*}}:2 = fork [2] %{{.*}} : none
// CHECK-NEXT:    %{{.*}}:2 = unpack %{{.*}} : tuple<i512, i1>
// CHECK-NEXT:    %{{.*}}:2 = fork [2] %{{.*}} : i1
// CHECK-NEXT:    %{{.*}}:2 = fork [2] %{{.*}} : i512
// CHECK-NEXT:    %{{.*}} = merge %{{.*}}#0 : i512
// CHECK-NEXT:    %{{.*}} = constant %{{.*}}#0 {value = 0 : i512} : i512
// CHECK-NEXT:    %{{.*}} = arith.cmpi sgt, %{{.*}}, %{{.*}} : i512
// CHECK-NEXT:    %{{.*}} = pack %{{.*}}#1, %{{.*}}#1 : tuple<i512, i1>
// CHECK-NEXT:    %{{.*}} = arith.ori %{{.*}}, %{{.*}}#0 : i1
// CHECK-NEXT:    %{{.*}}:2 = fork [2] %{{.*}} : i1
// CHECK-NEXT:    %{{.*}}, %{{.*}} = cond_br %{{.*}}#1, %{{.*}} : tuple<i512, i1>
// CHECK-NEXT:    sink %{{.*}} : tuple<i512, i1>
// CHECK-NEXT:    %{{.*}}, %{{.*}} = cond_br %{{.*}}#0, %{{.*}}#1 : none
// CHECK-NEXT:    sink %{{.*}} : none
// CHECK-NEXT:    return %{{.*}}, %{{.*}}, %{{.*}} : tuple<i512, i1>, none, none
// CHECK-NEXT:  }
// CHECK-NEXT:  handshake.func private @[[LABEL_MAP:.*]](%{{.*}}: tuple<i512, i1>, %{{.*}}: none, %{{.*}}, ...) -> (tuple<i512, i1>, none, none)
// CHECK-NEXT:    %{{.*}}:2 = fork [2] %{{.*}} : none
// CHECK-NEXT:    %{{.*}}:2 = unpack %{{.*}} : tuple<i512, i1>
// CHECK-NEXT:    %{{.*}} = merge %{{.*}} : i512
// CHECK-NEXT:    %{{.*}} = constant %{{.*}}#0 {value = 42 : i512} : i512
// CHECK-NEXT:    %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i512
// CHECK-NEXT:    %{{.*}} = pack %{{.*}}, %{{.*}} : tuple<i512, i1>
// CHECK-NEXT:    return %{{.*}}, %{{.*}}#1, %{{.*}} : tuple<i512, i1>, none, none
// CHECK-NEXT:  }
// CHECK-NEXT:  handshake.func @combined(%{{.*}}: tuple<i512, i1>, %{{.*}}: none, %{{.*}}: none, ...) -> (tuple<i512, i1>, none, none)
// CHECK-NEXT:    %{{.*}}:3 = instance @[[LABEL_MAP]](%{{.*}}, %{{.*}}, %{{.*}}) : (tuple<i512, i1>, none, none) -> (tuple<i512, i1>, none, none)
// CHECK-NEXT:    %{{.*}}:3 = instance @[[LABEL_FILTER]](%{{.*}}#0, %{{.*}}#1, %{{.*}}#2) : (tuple<i512, i1>, none, none) -> (tuple<i512, i1>, none, none)
// CHECK-NEXT:    return %{{.*}}#0, %{{.*}}#1, %{{.*}}#2 : tuple<i512, i1>, none, none
// CHECK-NEXT:  }
