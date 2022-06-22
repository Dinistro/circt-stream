module driver(
  input clock,
  input reset
);
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

  logic [1:0] state = 0;

  logic s0_done = 0;
  logic s1_done = 0;

  always @(posedge clock) begin
    if(reset == 1) begin
      inCtrl_valid = 1;
      // not yet ready ro receive results
      out0_ready = 0;
      out1_ready = 0;
      out2_ready = 0;
      out3_ready = 0;

      outCtrl_ready = 1;

      state = 1;
    end
    else if(state == 1) begin
      // holds valid high for 1 cycle.
      state = 2;
    end
    else if(state == 2) begin
      // sets valid to 0 to make sure the ctrl signal was only fired once.
      inCtrl_valid = 0;

      // now ready to receive elements
      out0_ready = 1;
      out1_ready = 1;
      out2_ready = 1;
      out3_ready = 1;

      state = 3;
    end
    begin
      if(out0_valid == 1 && s0_done == 0) begin
        if(out0_data_field1 == 0) begin
          $display("S0: Element=%d", out0_data_field0);
        end
        else begin
          $display("S0: EOS");
          s0_done = 1;
        end
      end
      if(out2_valid == 1 && s1_done == 0) begin
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
  end
endmodule // driver
