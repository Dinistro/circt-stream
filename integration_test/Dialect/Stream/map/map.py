import cocotb
from helper import initDut, Stream


@cocotb.test()
async def ascendingInputs(dut):
  ins, outs = await initDut(dut)

  in0 = Stream(ins[0], ins[1])
  out0 = Stream(outs[0], outs[1])

  out0Check = cocotb.start_soon(out0.checkOutputs([11,12,13]))

  cocotb.start_soon(in0.sendAndTerminate([1,2,3]))

  await out0Check

@cocotb.test()
async def multipleEOS(dut):
  ins, outs = await initDut(dut)

  in0 = Stream(ins[0], ins[1])
  out0 = Stream(outs[0], outs[1])

  for _ in range(5):
    out0Check = cocotb.start_soon(out0.checkOutputs([11,12,13]))
    sending = cocotb.start_soon(in0.sendAndTerminate([1,2,3]))

    await out0Check
