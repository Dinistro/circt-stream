// RUN: standalone-opt %s | standalone-opt | FileCheck %s

module {
  // CHECK-LABEL: func.func @bar()
  func.func @bar() {
    %0 = arith.constant 1 : i32
    // CHECK: %{{.*}} = standalone.foo %{{.*}} : i32
    %res = standalone.foo %0 : i32
    return
  }

  func.func @baz(%0 : i32) -> (i32) {
    // CHECK: %{{.*}} = standalone.neg_to_zero %{{.*}} : i32
    %res = standalone.neg_to_zero %0 : i32
    return %res : i32
  }
}
