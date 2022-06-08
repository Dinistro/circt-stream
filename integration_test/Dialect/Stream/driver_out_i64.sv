module driver();
  logic clock = 0;
  logic reset = 0;
  logic out0_valid, out0_ready;
  logic [63:0] out0_data_field0;
  logic out0_data_field1;
  logic inCtrl_valid, inCtrl_ready;
  logic outCtrl_valid, outCtrl_ready;

  top dut (.*);

  always begin
    // A clock period is #4.
    clock = ~clock;
    #2;
  end

  initial begin
    inCtrl_valid = 1;
    out0_ready = 1;
    outCtrl_ready = 1;

    reset = 1;
    // Hold reset high for one clock cycle.
    @(posedge clock);
    reset = 0;

    // Hold valid high for one clock cycle.
    @(posedge clock);
    inCtrl_valid = 0;
  end

  always @(posedge clock) begin
    if(out0_valid == 1) begin
      if(out0_data_field1 == 0) begin
        $display("Element=%d", out0_data_field0);
      end 
      else begin
        $display("EOS");
        $finish();
      end
    end
  end
endmodule // driver
