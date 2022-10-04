// RUN: stream-opt %s --convert-stream-to-handshake | FileCheck %s

module {
  func.func @sink(%in: !stream.stream<i32>) {
    stream.sink %in : !stream.stream<i32>
    return
  }

  // CHECK:      handshake.func @sink(%{{.*}}: tuple<i32, i1>, ...)
  // CHECK-NEXT:   sink %{{.*}} : tuple<i32, i1>
  // CHECK-NEXT:   return
  // CHECK-NEXT: }
}
