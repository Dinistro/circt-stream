// RUN: stream-opt %s --mlir-print-op-generic | stream-opt | FileCheck %s

module {
  func.func @memory_with_init(%in: !stream.stream<i32>) -> !stream.stream<i32> {
    %mem = stream.memory : memref<10xi32> {
    // For loop that fills up all the elements of the memref with 0
    ^0(%raw: memref<10xi32>):
      %0 = arith.constant 0 : index
      cf.br ^cond(%0: index)
    ^cond(%i: index):
      %c10 = arith.constant 10 : index
      %d = arith.cmpi eq, %c10, %i : index
      cf.cond_br %d, ^end, ^body
    ^body:
      %c0 = arith.constant 0 : i32
      memref.store %c0, %raw[%i] : memref<10xi32>
      %c1 = arith.constant 1 : index
      %ni = arith.addi %i, %c1 : index
      cf.br ^cond(%ni: index)
    ^end():
      stream.yield
    }
    %res = stream.map(%in, %mem) : (!stream.stream<i32>, memref<10xi32>) -> !stream.stream<i32> {
    ^0(%val : i32, %m: memref<10xi32>):
      %0 = arith.constant 1 : i32
      %r = arith.addi %0, %val : i32
      stream.yield %r : i32
    }
    return %res : !stream.stream<i32>
  }

}
