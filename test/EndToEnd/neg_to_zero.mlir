// RUN: stream-opt --convert-standalone-to-scf --convert-scf-to-cf --lower-std-to-handshake --lower-handshake-to-firrtl --cse --firrtl-lower-chirrtl --firrtl-infer-widths --firrtl-infer-resets --firrtl-dft --firrtl-prefix-modules --firrtl-lower-types --firrtl-expand-whens --canonicalize --firrtl-infer-rw --firrtl-inliner --firrtl-blackbox-reader --canonicalize -firrtl-remove-unused-ports --firrtl-emit-metadata --firrtl-emit-omir --lower-firrtl-to-hw --hw-memory-sim --cse --canonicalize --hw-cleanup --hw-legalize-modules --prettify-verilog --export-verilog %s | FileCheck %s

module {
  func.func @main(%0: i32) -> (i32) {
    %res = standalone.neg_to_zero %0 : i32
    return %res : i32
  }

  // TODO: how to test this? Use verilator, or should we simulate the handshake dialect?    

  // CHECK: hw.module @main(%in0_valid: i1, %in0_data: i32, %inCtrl_valid: i1, %out0_ready: i1, %outCtrl_ready: i1, %clock: i1, %reset: i1) -> (in0_ready: i1, inCtrl_ready: i1, out0_valid: i1, out0_data: i32, outCtrl_valid: i1) {
}
