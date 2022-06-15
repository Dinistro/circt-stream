// RUN: stream-opt %s --convert-stream-to-handshake | FileCheck %s

module {
  func.func @sink(%in: !stream.stream<i32>) {
    stream.sink %in : !stream.stream<i32>
    return
  }

  // CHECK:      handshake.func private @[[LABEL:.*]](%{{.*}}: tuple<i32, i1>, %{{.*}}: none, %{{.*}}: none, ...) -> none
  // CHECK-NEXT:   sink %{{.*}} : none
  // CHECK-NEXT:   sink %{{.*}} : tuple<i32, i1>
  // CHECK-NEXT:   return %{{.*}} : none
  // CHECK-NEXT: }
  // CHECK-NEXT: handshake.func @sink(%{{.*}}: tuple<i32, i1>, %{{.*}}: none, %{{.*}}: none, ...) -> none
  // CHECK-NEXT:   %{{.*}}:2 = fork [2] %{{.*}} : none
  // CHECK-NEXT:   %{{.*}} = instance @[[LABEL]](%{{.*}}, %{{.*}}, %{{.*}}#1) : (tuple<i32, i1>, none, none) -> none
  // CHECK-NEXT:   sink %{{.*}} : none
  // CHECK-NEXT:   return %{{.*}}#0 : none
  // CHECK-NEXT: }
}
