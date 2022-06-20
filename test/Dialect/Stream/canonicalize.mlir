// RUN: stream-opt %s -canonicalize -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL:   func.func @pack_unpack(
// CHECK-SAME:                      %[[a:.*]]: i32,
// CHECK-SAME:                      %[[b:.*]]: i64) -> i32 {
// CHECK:           return %[[a]] : i32
// CHECK:         }
func.func @pack_unpack(%a: i32, %b: i64) -> i32 {
  %res = stream.pack %a, %b : tuple<i32, i64>
  %a2, %b2 = stream.unpack %res : tuple<i32, i64>
  return %a2 : i32
}

// CHECK-LABEL:   func.func @pack_unpack2(
// CHECK-SAME:                       %[[a:.*]]: i32,
// CHECK-SAME:                       %[[b:.*]]: i64) -> tuple<i32, i64> {
// CHECK:           %[[res:.*]] = stream.pack %[[a]], %[[b]] : tuple<i32, i64>
// CHECK:           return %[[res]] : tuple<i32, i64>
// CHECK:         }
func.func @pack_unpack2(%a: i32, %b: i64) -> tuple<i32, i64> {
  %res = stream.pack %a, %b : tuple<i32, i64>
  %a2, %b2 = stream.unpack %res : tuple<i32, i64>
  return %res : tuple<i32, i64>
}

// CHECK-LABEL:   func.func @pack_unpack_negative(
// CHECK-SAME:                                    %[[a:.*]]: i32,
// CHECK-SAME:                                    %[[other_res:.*]]: tuple<i32, i64>) {
// CHECK:           %[[res:.*]] = stream.pack %[[a]], %[[a]] : tuple<i32, i32>
// CHECK:           %[[ab:.*]]:2 = stream.unpack %[[other_res]] : tuple<i32, i64>
// CHECK:           "test.foo"(%[[res]], %[[ab]]#0, %[[ab]]#1) : (tuple<i32, i32>, i32, i64) -> ()
// CHECK:           return
// CHECK:         }

func.func @pack_unpack_negative(%a: i32, %other_res: tuple<i32, i64>) {
  %res = stream.pack %a, %a : tuple<i32, i32>
  %a2, %b2 = stream.unpack %other_res : tuple<i32, i64>
  "test.foo"(%res, %a2, %b2) : (tuple<i32, i32>, i32, i64) -> ()
  return
}

// CHECK-LABEL:   func.func @unpack_pack(
// CHECK-SAME:                           %[[res:.*]]: tuple<i32, i64>) -> tuple<i32, i64> {
// CHECK:           return %[[res]] : tuple<i32, i64>
// CHECK:         }
func.func @unpack_pack(%res: tuple<i32, i64>) -> tuple<i32, i64> {
  %a, %b = stream.unpack %res : tuple<i32, i64>
  %res2 = stream.pack %a, %b : tuple<i32, i64>
  return %res2 : tuple<i32, i64>
}

// CHECK-LABEL:   func.func @unpack_pack2(
// CHECK-SAME:                            %[[res:.*]]: tuple<i32, i64>) -> i32 {
// CHECK:           %[[a:.*]]:2 = stream.unpack %[[res]] : tuple<i32, i64>
// CHECK:           return %[[a]]#0 : i32
// CHECK:         }
func.func @unpack_pack2(%res: tuple<i32, i64>) -> i32 {
  %a, %b = stream.unpack %res : tuple<i32, i64>
  %res2 = stream.pack %a, %b : tuple<i32, i64>
  return %a : i32
}

// CHECK-LABEL:   func.func @unpack_pack_negative(
// CHECK-SAME:                                    %[[arg:.*]]: i32,
// CHECK-SAME:                                    %[[other_res:.*]]: tuple<i32, i64>) {
// CHECK:           %[[ab:.*]]:2 = stream.unpack %[[other_res]] : tuple<i32, i64>
// CHECK:           %[[res:.*]] = stream.pack %[[arg]], %[[arg]] : tuple<i32, i32>
// CHECK:           "test.foo"(%[[res]], %[[ab]]#0, %[[ab]]#1) : (tuple<i32, i32>, i32, i64) -> ()
// CHECK:           return
// CHECK:         }

func.func @unpack_pack_negative(%arg: i32, %other_res: tuple<i32, i64>) {
  %a, %b = stream.unpack %other_res : tuple<i32, i64>
  %res = stream.pack %arg, %arg : tuple<i32, i32>
  "test.foo"(%res, %a, %b) : (tuple<i32, i32>, i32, i64) -> ()
  return
}
