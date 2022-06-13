module driver();
  logic clock = 0;
  logic reset = 0;

  logic out0_valid, out0_ready;
  logic [63:0] out0_data_field0;
  logic out0_data_field1;
  logic out1_valid, out1_ready;

  logic out2_valid, out2_ready;
  logic [63:0] out2_data_field0;
  logic out2_data_field1;
  logic out3_valid, out3_ready;

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
    out1_ready = 1;
    out2_ready = 1;
    out3_ready = 1;
    outCtrl_ready = 1;

    reset = 1;
    // Hold reset high for one clock cycle.
    @(posedge clock);
    reset = 0;

    // Hold valid high for one clock cycle.
    @(posedge clock);
    inCtrl_valid = 0;
  end

  logic s0_done = 0;
  logic s1_done = 0;

  always @(posedge clock) begin
    if(out0_valid == 1 && s0_done == 0) begin
      if(out0_data_field1 == 0) begin
        $display("S0: Element=%d", out0_data_field0);
      end
      else begin
        $display("S0: EOS");
        s0_done = 1;
      end
    end
    if(out1_valid == 1 && s1_done == 0) begin
      if(out2_data_field1 == 0) begin
        $display("S1: Element=%d", out2_data_field0);
      end
      else begin
        $display("S1: EOS");
        s1_done = 1;
      end
    end

    if(s0_done == 1 && s1_done == 1) begin
      $finish();
    end
  end
endmodule // driver
