// RUN: stream-opt %s --convert-stream-to-handshake --split-input-file | FileCheck %s

func.func @noop(%in: !stream.stream<i32>) -> !stream.stream<i32> {
  return %in : !stream.stream<i32>
}
// CHECK-LABEL: handshake.func @noop(
// CHECK-SAME:  %[[VAL:.*]]: tuple<i32, i1>, ...) -> tuple<i32, i1>
// CHECK-NEXT:    return %[[VAL]] : tuple<i32, i1>
// CHECK-NEXT:  }

// -----

func.func @noop_multi_stream(%in0: !stream.stream<i32>, %in1: !stream.stream<i64>) -> (!stream.stream<i32>, !stream.stream<i64>) {
  return %in0, %in1 : !stream.stream<i32>, !stream.stream<i64>
}

// CHECK-LABEL: handshake.func @noop_multi_stream(
// CHECK-SAME:      %{{.*}}: tuple<i32, i1>,
// CHECK-SAME:      %{{.*}}: tuple<i64, i1>, ...) ->
// CHECK-SAME:        (tuple<i32, i1>, tuple<i64, i1>)
// CHECK-NEXT:   return %{{.*}}, %{{.*}} : tuple<i32, i1>, tuple<i64, i1>
// CHECK-NEXT: }
