import cocotb
from helper import initDut, Stream


@cocotb.test()
async def ascendingInputs(dut):
  ins, outs = await initDut(dut)

  in0 = Stream(ins[0], ins[1])
  out0 = Stream(outs[0], outs[1])

  out0Check = cocotb.start_soon(out0.checkOutputs([1,2,4]))

  cocotb.start_soon(in0.sendAndTerminate([0,1,2,0,4,0]))

  await out0Check
