// RUN: standalone-opt %s | standalone-opt | FileCheck %s

module {
    func @min_op(%in: !stream.stream<i32>) {
        %min = stream.min(%in) : (!stream.stream<i32>) -> i32
        return
    }

    // CHECK: func @min_op(%{{.*}}: !stream.stream<i32>) {
}
