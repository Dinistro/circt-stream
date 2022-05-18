// RUN: standalone-opt %s --mlir-print-op-generic | standalone-opt | FileCheck %s

module {
    func.func @min_op(%in: !stream.stream<i32>) {
        %min = stream.min(%in) : (!stream.stream<i32>) -> i32
        return
    }

    // CHECK: func.func @min_op(%{{.*}}: !stream.stream<i32>) {
    // CHECK-NEXT:   %{{.*}} = stream.min(%{{.*}}) : (!stream.stream<i32>) -> i32

    func.func @min_cont(%in: !stream.stream<i32>) -> !stream.stream<i32> {
        %res = stream.min_continuous(%in) : (!stream.stream<i32>) -> !stream.stream<i32>
        return %res : !stream.stream<i32>
    }

    // CHECK: func.func @min_cont(%{{.*}}: !stream.stream<i32>) -> !stream.stream<i32> {
    // CHECK-NEXT:    %{{.*}} = stream.min_continuous(%{{.*}}) : (!stream.stream<i32>) -> !stream.stream<i32>
    // CHECK-NEXT:    return %{{.*}} : !stream.stream<i32>

    func.func @map(%in: !stream.stream<i32>) -> !stream.stream<i32> {
        %res = stream.map(%in) : (!stream.stream<i32>) -> !stream.stream<i32> {
        ^0(%val : i32):
            %0 = arith.constant 1 : i32
            %r = arith.addi %0, %val : i32
            stream.yield %r : i32
        }
        return %res : !stream.stream<i32>
    }

    // CHECK: func.func @map(%{{.*}}: !stream.stream<i32>) -> !stream.stream<i32> {
    // CHECK-NEXT:    %{{.*}} = stream.map(%{{.*}}) : (!stream.stream<i32>) -> !stream.stream<i32> {
    // CHECK-NEXT:    ^{{.*}}(%{{.*}}: i32):
    // CHECK-NEXT:        %{{.*}} = arith.constant 1 : i32
    // CHECK-NEXT:        %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
    // CHECK-NEXT:        stream.yield %{{.*}} : i32
    // CHECK-NEXT:    }
    // CHECK-NEXT:    return %{{.*}} : !stream.stream<i32>
    // CHECK-NEXT: }

    func.func @filter(%in: !stream.stream<i32>) -> !stream.stream<i32> {
      %res = stream.filter(%in) : (!stream.stream<i32>) -> !stream.stream<i32> {
      ^0(%val : i32):
        %cond = arith.constant false
        stream.yield %cond : i1
      }
      return %res : !stream.stream<i32>
    }

    // CHECK: func.func @filter(%{{.*}}: !stream.stream<i32>) -> !stream.stream<i32> {
    // CHECK-NEXT:     %{{.*}} = stream.filter(%{{.*}}) : (!stream.stream<i32>) -> !stream.stream<i32> {
    // CHECK-NEXT:     ^{{.*}}(%{{.*}}: i32):
    // CHECK-NEXT:         %{{.*}} = arith.constant false
    // CHECK-NEXT:         stream.yield %{{.*}} : i1
    // CHECK-NEXT:     }
    // CHECK-NEXT:     return %{{.*}} : !stream.stream<i32>
    // CHECK-NEXT: }

  func.func @reduce(%in: !stream.stream<i64>) -> !stream.stream<i64> {
    %res = stream.reduce(%in) {initValue = 0 : i64}: (!stream.stream<i64>) -> !stream.stream<i64> {
    ^0(%acc: i64, %val: i64):
      %r = arith.addi %acc, %val : i64
      stream.yield %r : i64
    }
    return %res : !stream.stream<i64>
  }

  // CHECK: func.func @reduce(%{{.*}}: !stream.stream<i64>) -> !stream.stream<i64> {
  // CHECK-NEXT:  %{{.*}} = stream.reduce(%a{{.*}}) {initValue = 0 : i64} : (!stream.stream<i64>) -> !stream.stream<i64> {
  // CHECK-NEXT:  ^bb0(%{{.*}}: i64, %{{.*}}: i64):
  // CHECK-NEXT:    %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i64
  // CHECK-NEXT:    stream.yield %{{.*}} : i64
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return %{{.*}} : !stream.stream<i64>
  // CHECK-NEXT:}

  func.func @tuples(%tuple: tuple<i32, i64>) -> tuple<i64, i32> {
    %a, %b = stream.unpack %tuple : tuple<i32, i64>
    %res = stream.pack %b, %a : tuple<i64, i32>
    return %res : tuple<i64, i32>
  }

  // CHECK: func.func @tuples(%{{.*}}: tuple<i32, i64>) -> tuple<i64, i32> {
  // CHECK-NEXT:  %{{.*}}:2 = stream.unpack %{{.*}} : tuple<i32, i64>
  // CHECK-NEXT:  %{{.*}} = stream.pack %{{.*}}#1, %{{.*}}#0 : tuple<i64, i32>
  // CHECK-NEXT:  return %{{.*}} : tuple<i64, i32>
  // CHECK-NEXT:}

}
