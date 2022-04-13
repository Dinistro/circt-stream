// RUN: standalone-opt %s | standalone-opt | FileCheck %s

module {
    func @type(%in: !stream.stream<i32>) {
        return
    }

    // CHECK: func @type(%{{.*}}: !stream.stream<i32>) {
}
