import cocotb
from helper import initDut, Stream
import random


async def initStreams(dut):
  ins, outs = await initDut(dut)

  in0 = Stream(ins[0])
  out0 = Stream(outs[0])
  return in0, out0

@cocotb.test()
async def increase(dut):
  inStream, outStream = await initStreams(dut)

  resCheck = cocotb.start_soon(outStream.checkOutputs([(2, 2), (4, 4),
                                                       (6, 6)]))

  cocotb.start_soon(inStream.sendAndTerminate([(1, 1), (2, 2), (3, 3)]))

  await resCheck


@cocotb.test()
async def mixed(dut):
  inStream, outStream = await initStreams(dut)

  resCheck = cocotb.start_soon(
      outStream.checkOutputs([(11, 11), (4, 11), (30, 30), (29, 30)]))

  inputs = [(1, 10), (2, 2), (30, 0), (15, 14)]
  cocotb.start_soon(inStream.sendAndTerminate(inputs))

  await resCheck


@cocotb.test()
async def randomized(dut):
  inStream, outStream = await initStreams(dut)

  N = 100

  inputs = [(random.randint(0, 1000), random.randint(0, 1000))
            for _ in range(N)]
  outputs = []
  m = 0
  for (l, r) in inputs:
    s = l + r
    m = max(s, m)
    outputs.append((s, m))

  resCheck = cocotb.start_soon(outStream.checkOutputs(outputs))
  cocotb.start_soon(inStream.sendAndTerminate(inputs))

  await resCheck
