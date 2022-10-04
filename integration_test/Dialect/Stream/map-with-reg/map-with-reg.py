import cocotb
from helper import initDut, Stream


@cocotb.test()
async def ascendingInputs(dut):
  ins, outs = await initDut(dut)

  in0 = Stream(ins[0])
  out0 = Stream(outs[0])

  out0Check = cocotb.start_soon(out0.checkOutputs([1,3,6]))

  cocotb.start_soon(in0.sendAndTerminate([1,2,3]))

  await out0Check

@cocotb.test()
async def multipleEOS(dut):
  ins, outs = await initDut(dut)

  in0 = Stream(ins[0])
  out0 = Stream(outs[0])

  for _ in range(5):
    out0Check = cocotb.start_soon(out0.checkOutputs([1,3,6]))

    cocotb.start_soon(in0.sendAndTerminate([1,2,3]))

    await out0Check
