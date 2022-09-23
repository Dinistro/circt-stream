import cocotb
import cocotb.clock
from cocotb.triggers import RisingEdge, ReadWrite
import re


#TODO reuse parts of CIRCT's cocotb helper
class HandshakePort:
  """
  Helper class that encapsulates a handshake port from the DUT.
  """

  def __init__(self, dut, rdy, val):
    self.dut = dut
    self.ready = rdy
    self.valid = val

  def isReady(self):
    return self.ready.value.is_resolvable and self.ready.value == 1

  def setReady(self, v):
    self.ready.value = v

  def isValid(self):
    return self.valid.value.is_resolvable and self.valid.value == 1

  def isCtrl(self):
    return True

  def setValid(self, v):
    self.valid.value = v

  async def waitUntilReady(self):
    while (not self.isReady()):
      await RisingEdge(self.dut.clock)

  async def waitUntilValid(self):
    while (not self.isValid()):
      await RisingEdge(self.dut.clock)

  async def awaitHandshake(self):
    # Make sure that changes to ready are propagated before it is checked.
    await ReadWrite()
    directSend = self.isReady()
    await self.waitUntilReady()

    if (directSend):
      # If it was initially ready, the handshake happens in the current cycle.
      # Thus the invalidation has to wait until the next cycle
      await RisingEdge(self.dut.clock)

    self.setValid(0)

    if (not directSend):
      # The handshake happend already, so we only have to ensure that valid 0
      # gets communicated correctly.
      await RisingEdge(self.dut.clock)

  async def send(self, val=None):
    self.setValid(1)
    await self.awaitHandshake()

  async def awaitNOutputs(self, n):
    assert (self.isReady())
    for _ in range(n):
      await self.waitUntilValid()
      await RisingEdge(self.dut.clock)


class HandshakeDataPort(HandshakePort):
  """
  A handshaked port with a data field.
  """

  def __init__(self, dut, rdy, val, data):
    super().__init__(dut, rdy, val)
    self.data = data

  def isCtrl(self):
    return False

  async def send(self, val):
    self.data.value = val
    await super().send()

  async def checkOutputs(self, results):
    assert (self.isReady())
    for res in results:
      await self.waitUntilValid()
      assert (self.data.value == res)
      await RisingEdge(self.dut.clock)

  async def collectNOutputs(self, n):
    assert (self.isReady())
    res = []
    for _ in range(n):
      await self.waitUntilValid()
      res.append(self.data.value.integer)
      await RisingEdge(self.dut.clock)
    return res

  async def collectUntil(self, checkFunc):
    assert (self.isReady())
    res = []
    while True:
      await self.waitUntilValid()
      val = self.data.value.integer
      res.append(val)
      await RisingEdge(self.dut.clock)
      if (checkFunc(res, val)):
        break
    return res


class HandshakeTuplePort(HandshakePort):
  """
  A handshaked port that sends a tuple.
  """

  def __init__(self, dut, rdy, val, fields):
    super().__init__(dut, rdy, val)
    self.fields = fields

  def isCtrl(self):
    return False

  def _assignTupleValue(self, val, curr):
    assert (len(list(val)) == len(curr))
    for (f, v) in zip(curr, list(val)):
      if (isinstance(f, list)):
        assert isinstance(v, tuple)
        self._assignTupleValue(v, f)
      else:
        assert not isinstance(v, tuple)
        f.value = v

  async def send(self, val):
    self._assignTupleValue(val, self.fields)

    await super().send()

  def _collectValRec(self, curr):
    if not isinstance(curr, list):
      return curr.value.integer
    return tuple([self._collectValRec(f) for f in curr])

  async def checkOutputs(self, results):
    assert (self.isReady())
    for res in results:
      await self.waitUntilValid()
      val = self._collectValRec(self.fields)
      assert (res == val)

      await RisingEdge(self.dut.clock)

  async def collectNOutputs(self, n):
    assert (self.isReady())
    res = []
    for _ in range(n):
      await self.waitUntilValid()
      t = self._collectValRec(self.fields)
      res.append(t)
      await RisingEdge(self.dut.clock)
    return res

  async def collectUntil(self, checkFunc):
    assert (self.isReady())
    res = []
    while True:
      await self.waitUntilValid()
      t = tuple([f.value.integer for f in self.fields])
      res.append(t)
      await RisingEdge(self.dut.clock)
      if (checkFunc(res, t)):
        break
    return res


def buildTupleStructure(dut, tupleFields, prefix):
  """
  Helper that builds a neasted list structure that represents the nester tuple

structure of the inputs.
  """
  size = 0
  while True:
    r = re.compile(f"{prefix}_field{size}")
    found = False
    for f in tupleFields:
      if r.match(f):
        found = True
        break

    if not found:
      break
    size += 1

  res = []
  for i in range(size):
    fName = f"{prefix}_field{i}"
    if fName in tupleFields:
      res.append(getattr(dut, fName))
      continue

    res.append(buildTupleStructure(dut, tupleFields, fName))

  return res


def _findPort(dut, name):
  """
  Checks if dut has a port of the provided name. Either throws an exception or
  returns a HandshakePort that encapsulates the dut's interface.
  """
  readyName = f"{name}_ready"
  validName = f"{name}_valid"
  dataName = f"{name}_data"
  if (not hasattr(dut, readyName) or not hasattr(dut, validName)):
    raise Exception(f"dut does not have a port named {name}")

  ready = getattr(dut, readyName)
  valid = getattr(dut, validName)
  data = getattr(dut, dataName, None)

  # Needed, as it otherwise would try to resolve the value
  hasData = not isinstance(data, type(None))
  if hasData:
    return HandshakeDataPort(dut, ready, valid, data)

  isCtrl = not hasattr(dut, f"{name}_data_field0")

  r = re.compile(f"^{name}_data_field")
  tupleFields = [f for f in dir(dut) if r.match(f)]
  isCtrl = not any(tupleFields)

  if (isCtrl):
    return HandshakePort(dut, ready, valid)

  fields = buildTupleStructure(dut, tupleFields, f"{name}_data")

  return HandshakeTuplePort(dut, ready, valid, fields)


def getPorts(dut, inNames, outNames):
  """
  Helper function to produce in and out ports for the provided dut.
  """
  ins = [_findPort(dut, name) for name in inNames]
  outs = [_findPort(dut, name) for name in outNames]
  return ins, outs


def getNames(dut, prefix):
  names = []

  i = 0
  while hasattr(dut, f"{prefix}{i}_ready"):
    names.append(f"{prefix}{i}")
    i += 1

  return names


def getInNames(dut):
  return getNames(dut, "in")


def getOutNames(dut):
  return getNames(dut, "out")


async def initDut(dut, inNames=None, outNames=None):
  """
  Initializes a dut by adding a clock, setting initial valid and ready flags,
  and performing a reset.
  """
  if (inNames is None):
    inNames = getInNames(dut)

  if (outNames is None):
    outNames = getOutNames(dut)

  ins, outs = getPorts(dut, inNames, outNames)

  # Create a 10us period clock on port clock
  clock = cocotb.clock.Clock(dut.clock, 10, units="us")
  cocotb.start_soon(clock.start())  # Start the clock

  for i in ins:
    i.setValid(0)

  for o in outs:
    o.setReady(1)

  # Reset
  dut.reset.value = 1
  await RisingEdge(dut.clock)
  dut.reset.value = 0
  await RisingEdge(dut.clock)
  return ins, outs


class Stream:
  """
  Class that encapsulates all the handshake ports for a stream
  """

  def __init__(self, dataPort, ctrlPort):
    self.dataPort = dataPort
    self.ctrlPort = ctrlPort

  async def sendData(self, data):
    ds = cocotb.start_soon(self.dataPort.send((data, 0)))
    cs = cocotb.start_soon(self.ctrlPort.send())
    await ds
    await cs

  def _buildSentinel(self, fields):
    if not isinstance(fields, list):
      return 0
    return tuple([self._buildSentinel(f) for f in fields])

  async def sendEOS(self):
    data = cocotb.start_soon(
        self.dataPort.send((self._buildSentinel(self.dataPort.fields[0]), 1)))
    ctrl = cocotb.start_soon(self.ctrlPort.send())
    await data
    await ctrl

  async def sendAndTerminate(self, data):
    for d in data:
      await self.sendData(d)

    await self.sendEOS()

  async def checkOutputs(self, results):
    resWithEOS = [(d, 0) for d in results]
    data = cocotb.start_soon(self.dataPort.checkOutputs(resWithEOS))
    ctrl = cocotb.start_soon(self.ctrlPort.awaitNOutputs(len(resWithEOS) + 1))
    await data
    [(_, eos)] = await cocotb.start_soon(self.dataPort.collectNOutputs(1))
    assert eos == 1

    await ctrl
