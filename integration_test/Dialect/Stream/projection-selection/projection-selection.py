import cocotb
from helper import initDut, Stream
import random


@cocotb.test()
async def single(dut):
  ins, outs = await initDut(dut)

  in0 = Stream(ins[0])
  out0 = Stream(outs[0])

  out0Check = cocotb.start_soon(out0.checkOutputs([(3, 11)]))

  cocotb.start_soon(in0.sendAndTerminate([(1, 2, 3, 4, 5, 6, 7, 8)]))

  await out0Check


def randomTuple():
  return tuple([random.randint(0, 100) for _ in range(8)])

def getOutpus(inputs):
  asLists = [list(i) for i in inputs]
  return [(l,r) for i in asLists if (l := i[0] + i[1]) <= (r := i[4] + i[5])]


@cocotb.test()
async def multiple(dut):
  ins, outs = await initDut(dut)

  in0 = Stream(ins[0])
  out0 = Stream(outs[0])

  N = 100
  inputs = [randomTuple() for _ in range(N)]
  outputs = getOutpus(inputs)

  out0Check = cocotb.start_soon(out0.checkOutputs(outputs))
  cocotb.start_soon(in0.sendAndTerminate(inputs))

  await out0Check

@cocotb.test()
async def multipleEOS(dut):
  ins, outs = await initDut(dut)

  in0 = Stream(ins[0])
  out0 = Stream(outs[0])

  for _ in range(3):
    N = 10
    inputs = [randomTuple() for _ in range(N)]
    outputs = getOutpus(inputs)

    out0Check = cocotb.start_soon(out0.checkOutputs(outputs))
    cocotb.start_soon(in0.sendAndTerminate(inputs))

    await out0Check
