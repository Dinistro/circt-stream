import cocotb
import cocotb.clock
from cocotb.triggers import FallingEdge, RisingEdge

# Hack to allow imports from parent directory
import sys
import os
from helper import HandshakePort, getPorts


@cocotb.test()
async def singleOut(dut):
  [inCtrl], [out0, out1, outCtrl] = getPorts(dut, ["inCtrl"],
                                             ["out0", "out1", "outCtrl"])

  # Create a 10us period clock on port clock
  clock = cocotb.clock.Clock(dut.clock, 10, units="us")
  cocotb.start_soon(clock.start())  # Start the clock

  inCtrl.setValid(0)

  out0.setReady(1)
  out1.setReady(1)
  outCtrl.setReady(1)

  # Reset
  dut.reset.value = 1
  await RisingEdge(dut.clock)
  dut.reset.value = 0
  await RisingEdge(dut.clock)

  # Collect until EOS
  resCollect = cocotb.start_soon(
      out0.collectUntil(lambda l, t: list(t)[-1] == 1))

  # Just send on ctrl signal, the create will ensure the stream creation
  inCtrlSend = cocotb.start_soon(inCtrl.send())
  await inCtrlSend

  res = await resCollect
  for (data, eos) in res:
    if (eos == 0):
      print(f"Element={data}")
    else:
      print("EOS")


def getOutNames(dut):
  names = []

  i = 0
  while hasattr(dut, f"out{i}_ready"):
    names.append(f"out{i}")
    i += 1

  names.append(f"outCtrl")
  return names


@cocotb.test()
async def multipleOut(dut):
  outNames = getOutNames(dut)
  [inCtrl], outs = getPorts(dut, ["inCtrl"], outNames)

  # Create a 10us period clock on port clock
  clock = cocotb.clock.Clock(dut.clock, 10, units="us")
  cocotb.start_soon(clock.start())  # Start the clock

  inCtrl.setValid(0)

  for out in outs:
    out.setReady(1)

  # Drop the ctrl signals
  dataOuts = [out for out in outs if not out.isCtrl()]

  # Reset
  dut.reset.value = 1
  await RisingEdge(dut.clock)
  dut.reset.value = 0
  await RisingEdge(dut.clock)

  # Collect until EOS
  resAsync = [
      cocotb.start_soon(out.collectUntil(lambda l, t: list(t)[-1] == 1))
      for out in dataOuts
  ]

  # Just send on ctrl signal, the create will ensure the stream creation
  inCtrlSend = cocotb.start_soon(inCtrl.send())
  await inCtrlSend

  results = [await r for r in resAsync]
  for (i, res) in enumerate(results):
    for (data, eos) in res:
      if (eos == 0):
        print(f"S{i}: Element={data}")
      else:
        print(f"S{i}: EOS")
