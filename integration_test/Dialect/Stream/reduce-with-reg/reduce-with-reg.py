import cocotb
from helper import initDut, Stream


@cocotb.test()
async def ascendingInputs(dut):
  ins, outs = await initDut(dut)

  in0 = Stream(ins[0])
  out0 = Stream(outs[0])

  out0Check = cocotb.start_soon(out0.checkOutputs([36]))

  cocotb.start_soon(in0.sendAndTerminate([1,2,3]))

  await out0Check
