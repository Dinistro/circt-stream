// RUN: standalone-opt --convert-stream-to-handshake --canonicalize %s | FileCheck %s

module {
  func @noop(%in: !stream.stream<i32>) -> !stream.stream<i32> {
    return %in : !stream.stream<i32>
  }
  // CHECK: func @noop(%{{.*}}: i32) -> i32 { 
  // CHECK-NEXT:  return %{{.*}} : i32

  func @map(%in: !stream.stream<i32>) -> !stream.stream<i32> {
    %res = stream.map(%in) : (!stream.stream<i32>) -> !stream.stream<i32> {
    ^0(%val : i32):
      %0 = arith.constant 1 : i32
      %r = arith.addi %0, %val : i32
      stream.yield %r : i32
    }
    return %res : !stream.stream<i32>
  }
  // CHECK: func @map(%{{.*}}: i32) -> i32 {
  // CHECK-NEXT:  %{{.*}} = arith.constant 1 : i32
  // CHECK-NEXT:  %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
  // CHECK-NEXT:  return %{{.*}} : i32

}
