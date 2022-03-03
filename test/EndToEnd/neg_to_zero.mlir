// RUN: standalone-opt --convert-standalone-to-scf --convert-scf-to-cf --lower-std-to-handshake --lower-handshake-to-firrtl --cse --firrtl-lower-chirrtl --firrtl-infer-widths --firrtl-infer-resets --firrtl-dft --firrtl-prefix-modules --firrtl-lower-types --firrtl-expand-whens --canonicalize --firrtl-infer-rw --firrtl-inliner --firrtl-blackbox-reader --canonicalize -firrtl-remove-unused-ports --firrtl-emit-metadata --firrtl-emit-omir --lower-firrtl-to-hw --hw-memory-sim --cse --canonicalize --hw-cleanup --hw-legalize-modules --prettify-verilog --export-verilog %s | FileCheck %s

module {
    func @main(%0: i32) -> (i32) {
        %res = standalone.neg_to_zero %0 : i32
        return %res : i32
    }

  // TODO: how to test this? Use verilator, or should we simulate the handshake dialect?    


  // CHECK-LABEL: module main(	
  // CHECK-NEXT:    input         in0_valid,
  // CHECK-NEXT:    input  [31:0] in0_data,
  // CHECK-NEXT:    input         inCtrl_valid, out0_ready, outCtrl_ready, clock, reset,
  // CHECK-NEXT:    output        in0_ready, inCtrl_ready, out0_valid,
  // CHECK-NEXT:    output [31:0] out0_data,
  // CHECK-NEXT:    output        outCtrl_valid);
}
