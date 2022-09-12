import cocotb
from helper import initDut, Stream


@cocotb.test()
async def increase(dut):
  ins, outs = await initDut(dut)

  in0 = Stream(ins[0], ins[1])
  out0 = Stream(outs[0], outs[1])

  out0Check = cocotb.start_soon(out0.checkOutputs([2, 4, 6]))

  cocotb.start_soon(in0.sendAndTerminate([(1, 1), (2, 2), (3, 3)]))

  await out0Check
