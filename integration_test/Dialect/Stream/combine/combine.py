import cocotb
from helper import initDut, Stream


@cocotb.test()
async def ascendingInputs(dut):
  ins, outs = await initDut(dut)

  in0 = Stream(ins[0], ins[1])
  in1 = Stream(ins[2], ins[3])
  inCtrl = ins[-1]

  out0 = Stream(outs[0], outs[1])
  outCtrl = outs[-1]

  #init stream
  inCtrlSend = cocotb.start_soon(inCtrl.send())
  await inCtrlSend

  cocotb.start_soon(in0.sendAndTerminate([1, 2, 3]))
  cocotb.start_soon(in1.sendAndTerminate([10, 11, 12]))

  await cocotb.start_soon(out0.checkOutputs([11, 13, 15]))
