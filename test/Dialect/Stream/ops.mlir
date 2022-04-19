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

    //CHECK: func @min_cont(%{{.*}}: !stream.stream<i32>) -> !stream.stream<i32> {
    // CHECK-NEXT:    %{{.*}} = stream.min_continuous(%{{.*}}) : (!stream.stream<i32>) -> !stream.stream<i32>
    // CHECK-NEXT:    return %{{.*}} : !stream.stream<i32>

}
