import cocotb
import cocotb.clock
from cocotb.triggers import FallingEdge, RisingEdge

# Hack to allow imports from parent directory
import sys
import os
from helper import HandshakePort, getPorts, Stream
import random


async def initDut(dut):
  [in0, in1, inCtrl], [out0, out1,
                       outCtrl] = getPorts(dut, ["in0", "in1", "inCtrl"],
                                           ["out0", "out1", "outCtrl"])

  # Create a 10us period clock on port clock
  clock = cocotb.clock.Clock(dut.clock, 10, units="us")
  cocotb.start_soon(clock.start())  # Start the clock

  in0.setValid(0)
  in1.setValid(0)
  inCtrl.setValid(0)

  out0.setReady(1)
  out1.setReady(1)
  outCtrl.setReady(1)

  # Reset
  dut.reset.value = 1
  await RisingEdge(dut.clock)
  dut.reset.value = 0
  await RisingEdge(dut.clock)

  # init circuit
  inCtrlSend = cocotb.start_soon(inCtrl.send())
  await inCtrlSend

  inStream = Stream(in0, in1)
  outStream = Stream(out0, out1)
  return inStream, outStream


@cocotb.test()
async def increase(dut):
  inStream, outStream = await initDut(dut)

  resCheck = cocotb.start_soon(outStream.checkOutputs([(2, 2), (4, 4),
                                                       (6, 6)]))

  inputs = [(1, 1), (2, 2), (3, 3)]
  for i in inputs:
    await inStream.sendData(i)
  await inStream.sendEOS()

  await resCheck


@cocotb.test()
async def mixed(dut):
  inStream, outStream = await initDut(dut)

  resCheck = cocotb.start_soon(
      outStream.checkOutputs([(11, 11), (4, 11), (30, 30), (29, 30)]))

  inputs = [(1, 10), (2, 2), (30, 0), (15, 14)]
  for i in inputs:
    await inStream.sendData(i)
  await inStream.sendEOS()

  await resCheck


@cocotb.test()
async def randomized(dut):
  inStream, outStream = await initDut(dut)

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

  for i in inputs:
    await inStream.sendData(i)
  await inStream.sendEOS()

  await resCheck
