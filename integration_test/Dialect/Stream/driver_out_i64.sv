module driver(
  input clock,
  input reset
);
  logic out0_valid, out0_ready;
  logic out1_valid, out1_ready;
  logic [63:0] out0_data_field0;
  logic out0_data_field1;
  logic inCtrl_valid, inCtrl_ready;
  logic outCtrl_valid, outCtrl_ready;

  top dut (.*);

  logic [1:0] state = 0;

  always @(posedge clock) begin
    if(reset == 1) begin
      inCtrl_valid = 1;
      out0_ready = 1;
      out1_ready = 1;
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

      state = 3;
    end
    else begin
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
  end
endmodule // driver
