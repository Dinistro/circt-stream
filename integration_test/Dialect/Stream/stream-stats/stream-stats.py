import cocotb
from helper import initDut, Stream
import random


@cocotb.test()
async def sendOne(dut):
  ins, outs = await initDut(dut)

  in0 = Stream(ins[0])
  out0 = Stream(outs[0])
  out1 = Stream(outs[1])

  inputs = [(10, 2, 3, 6, 5, 1, 5, 9)]
  outputs = [max([max(list(i)) for i in inputs])]

  out0Check = cocotb.start_soon(out0.checkOutputs(inputs))
  out1Check = cocotb.start_soon(out1.checkOutputs(outputs))

  cocotb.start_soon(in0.sendAndTerminate(inputs))

  await out0Check
  await out1Check


def randomTuple():
  return tuple([random.randint(0, 100) for _ in range(8)])


@cocotb.test()
async def sendMultiple(dut):
  ins, outs = await initDut(dut)

  in0 = Stream(ins[0])
  out0 = Stream(outs[0])
  out1 = Stream(outs[1])

  N = 100
  inputs = [randomTuple() for _ in range(N)]
  outputs = [max([max(list(i)) for i in inputs])]

  out0Check = cocotb.start_soon(out0.checkOutputs(inputs))
  out1Check = cocotb.start_soon(out1.checkOutputs(outputs))

  cocotb.start_soon(in0.sendAndTerminate(inputs))

  await out0Check
  await out1Check


@cocotb.test()
async def sendMultipleWithEOS(dut):
  ins, outs = await initDut(dut)

  in0 = Stream(ins[0])
  out0 = Stream(outs[0])
  out1 = Stream(outs[1])

  for _ in range(5):
    N = 100
    inputs = [randomTuple() for _ in range(N)]
    outputs = [max([max(list(i)) for i in inputs])]

    out0Check = cocotb.start_soon(out0.checkOutputs(inputs))
    out1Check = cocotb.start_soon(out1.checkOutputs(outputs))

    cocotb.start_soon(in0.sendAndTerminate(inputs))

    await out0Check
    await out1Check
