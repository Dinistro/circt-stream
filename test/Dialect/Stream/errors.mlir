// RUN: standalone-opt %s --split-input-file --verify-diagnostics

func.func @create_wrong_type() {
  // expected-error @+1 {{element #1's type does not match the type of the stream: expected 'i32' got 'i64'}}
  %0 = "stream.create"() {values = [1 : i32, 2 : i64, 3 : i32, 4 : i32]} : () -> !stream.stream<i32>
}

// -----
