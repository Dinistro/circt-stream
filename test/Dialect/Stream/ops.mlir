// RUN: standalone-opt %s | standalone-opt | FileCheck %s

module {
    func @min_op(%in: !stream.stream<i32>) {
        %min = stream.min(%in) : (!stream.stream<i32>) -> i32
        return
    }

    // CHECK: func @min_op(%{{.*}}: !stream.stream<i32>) {
    // CHECK-NEXT:   %{{.*}} = stream.min(%{{.*}}) : (!stream.stream<i32>) -> i32

    func @min_cont(%in: !stream.stream<i32>) -> !stream.stream<i32> {
        %res = stream.min_continuous(%in) : (!stream.stream<i32>) -> !stream.stream<i32>
        return %res : !stream.stream<i32>
    }

    // CHECK: func @min_cont(%{{.*}}: !stream.stream<i32>) -> !stream.stream<i32> {
    // CHECK-NEXT:    %{{.*}} = stream.min_continuous(%{{.*}}) : (!stream.stream<i32>) -> !stream.stream<i32>
    // CHECK-NEXT:    return %{{.*}} : !stream.stream<i32>

    func @map(%in: !stream.stream<i32>) -> !stream.stream<i32> {
        %res = stream.map(%in) : (!stream.stream<i32>) -> !stream.stream<i32> {
        ^0(%val : i32):
            %0 = arith.constant 1 : i32
            %r = arith.addi %0, %val : i32
            stream.yield %r : i32
        }
        return %res : !stream.stream<i32>
    }

    // CHECK: func @map(%{{.*}}: !stream.stream<i32>) -> !stream.stream<i32> {
    // CHECK-NEXT:    %{{.*}} = stream.map(%{{.*}}) : (!stream.stream<i32>) -> !stream.stream<i32> {
    // CHECK-NEXT:    ^{{.*}}(%{{.*}}: i32):
    // CHECK-NEXT:        %{{.*}} = arith.constant 1 : i32
    // CHECK-NEXT:        %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
    // CHECK-NEXT:        stream.yield %{{.*}} : i32
    // CHECK-NEXT:    }
    // CHECK-NEXT:    return %{{.*}} : !stream.stream<i32>
    // CHECK-NEXT: }

    func @filter(%in: !stream.stream<i32>) -> !stream.stream<i32> {
      %res = stream.filter(%in) : (!stream.stream<i32>) -> !stream.stream<i32> {
      ^0(%val : i32):
        %cond = arith.constant false
        stream.yield %cond : i1
      }
      return %res : !stream.stream<i32>
    }

    // CHECK: func @filter(%{{.*}}: !stream.stream<i32>) -> !stream.stream<i32> {
    // CHECK-NEXT:     %{{.*}} = stream.filter(%{{.*}}) : (!stream.stream<i32>) -> !stream.stream<i32> {
    // CHECK-NEXT:     ^{{.*}}(%{{.*}}: i32):
    // CHECK-NEXT:         %{{.*}} = arith.constant false
    // CHECK-NEXT:         stream.yield %{{.*}} : i1
    // CHECK-NEXT:     }
    // CHECK-NEXT:     return %{{.*}} : !stream.stream<i32>
    // CHECK-NEXT: }

}
