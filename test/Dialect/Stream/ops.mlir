// RUN: stream-opt %s --mlir-print-op-generic | stream-opt | FileCheck %s

module {
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

  func.func @split(%in: !stream.stream<tuple<i32, i32>>) -> (!stream.stream<i32>, !stream.stream<i32>) {
    %res0, %res1 = stream.split(%in) : (!stream.stream<tuple<i32, i32>>) -> (!stream.stream<i32>, !stream.stream<i32>) {
    ^0(%val: tuple<i32, i32>):
      %0, %1 = stream.unpack %val : tuple<i32, i32>
      stream.yield %0, %1 : i32, i32
    }
    return %res0, %res1 : !stream.stream<i32>, !stream.stream<i32>
  }

  // CHECK: func.func @split(%{{.*}}: !stream.stream<tuple<i32, i32>>) -> (!stream.stream<i32>, !stream.stream<i32>) {
  // CHECK-NEXT:   %{{.*}}:2 = stream.split(%{{.*}}) : (!stream.stream<tuple<i32, i32>>) -> (!stream.stream<i32>, !stream.stream<i32>) {
  // CHECK-NEXT:   ^{{.*}}(%{{.*}}: tuple<i32, i32>):
  // CHECK-NEXT:     %{{.*}}:2 = stream.unpack %{{.*}} : tuple<i32, i32>
  // CHECK-NEXT:     stream.yield %{{.*}}#0, %{{.*}}#1 : i32, i32
  // CHECK-NEXT:   }
  // CHECK-NEXT:   return %{{.*}}#0, %{{.*}}#1 : !stream.stream<i32>, !stream.stream<i32>
  // CHECK-NEXT: }

  func.func @combine(%in0: !stream.stream<i32>, %in1: !stream.stream<i32>) -> (!stream.stream<tuple<i32, i32>>) {
    %res = stream.combine(%in0, %in1) : (!stream.stream<i32>, !stream.stream<i32>) -> (!stream.stream<tuple<i32, i32>>) {
    ^0(%val0: i32, %val1: i32):
      %0 = stream.pack %val0, %val1 : tuple<i32, i32>
      stream.yield %0 : tuple<i32, i32>
    }
    return %res : !stream.stream<tuple<i32, i32>>
  }

  // CHECK: func.func @combine(%{{.*}}: !stream.stream<i32>, %{{.*}}: !stream.stream<i32>) -> !stream.stream<tuple<i32, i32>> {
  // CHECK-NEXT:   %{{.*}} = stream.combine(%{{.*}}, %{{.*}}) : (!stream.stream<i32>, !stream.stream<i32>) -> !stream.stream<tuple<i32, i32>> {
  // CHECK-NEXT:   ^bb0(%{{.*}}: i32, %{{.*}}: i32):
  // CHECK-NEXT:     %{{.*}} = stream.pack %{{.*}}, %{{.*}} : tuple<i32, i32>
  // CHECK-NEXT:     stream.yield %{{.*}} : tuple<i32, i32>
  // CHECK-NEXT:   }
  // CHECK-NEXT:   return %{{.*}} : !stream.stream<tuple<i32, i32>>
  // CHECK-NEXT: }

  func.func @sink(%in: !stream.stream<i32>) {
    stream.sink %in : !stream.stream<i32>
    return
  }

  // CHECK:      func.func @sink(%{{.*}}: !stream.stream<i32>) {
  // CHECK-NEXT:   stream.sink %{{.*}} : !stream.stream<i32>
  // CHECK-NEXT:   return
  // CHECK-NEXT: }
}
