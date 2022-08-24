from cocotb.triggers import RisingEdge


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
      if(checkFunc(res, val)):
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

  async def send(self, val):
    assert (len(list(val)) == len(self.fields))
    for (f, v) in zip(self.fields, list(val)):
      f.value = v

    await super().send()

  async def checkOutputs(self, results):
    assert (self.isReady())
    for res in results:
      await self.waitUntilValid()

      assert (len(list(res)) == len(self.fields))
      for (f, r) in zip(self.fields, list(res)):
        assert (f.value == r)
      await RisingEdge(self.dut.clock)

  async def collectNOutputs(self, n):
    assert (self.isReady())
    res = []
    for _ in range(n):
      await self.waitUntilValid()
      t = tuple([f.value.integer for f in self.fields])
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
      if(checkFunc(res, t)):
        break
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

  if (isCtrl):
    return HandshakePort(dut, ready, valid)

  fields = []
  i = 0
  while hasattr(dut, f"{name}_data_field{i}"):
    fields.append(getattr(dut, f"{name}_data_field{i}"))
    i += 1

  return HandshakeTuplePort(dut, ready, valid, fields)


def getPorts(dut, inNames, outNames):
  """
  Helper function to produce in and out ports for the provided dut.
  """
  ins = [_findPort(dut, name) for name in inNames]
  outs = [_findPort(dut, name) for name in outNames]
  return ins, outs
