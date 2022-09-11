import cocotb
from helper import initDut, Stream


@cocotb.test()
async def ascendingInputs(dut):
  ins, outs = await initDut(dut)

  in0 = Stream(ins[0], ins[1])
  inCtrl = ins[-1]

  out0 = Stream(outs[0], outs[1])
  out1 = Stream(outs[2], outs[3])
  outCtrl = outs[-1]

  #init stream
  inCtrlSend = cocotb.start_soon(inCtrl.send())
  await inCtrlSend

  out0Check = cocotb.start_soon(out0.checkOutputs([1,2,3]))
  out1Check = cocotb.start_soon(out1.checkOutputs([1,3,5]))

  cocotb.start_soon(in0.sendAndTerminate([1,2,3]))

  await out0Check
  await out1Check
