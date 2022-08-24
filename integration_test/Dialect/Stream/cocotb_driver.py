import argparse
import os
import subprocess
import sys
import re
from pathlib import Path
from cocotb_test.simulator import run


def parseArgs(args):
  argparser = argparse.ArgumentParser(description="COCOTB driver for CIRCT")

  argparser.add_argument("--objdir",
                         type=str,
                         help="Select a directoy in which to run this test." +
                         " Must be different from other tests in the same" +
                         " directory. Defaults to 'sources[0].d'.")

  argparser.add_argument("--topLevel",
                         type=str,
                         help="Name of the top level verilog module.")

  argparser.add_argument("--simulator",
                         choices=['icarus'],
                         default="icarus",
                         help="Name of the simulator to use.")

  argparser.add_argument("--pythonModule",
                         type=str,
                         required=True,
                         help="Name of the python module.")

  argparser.add_argument("--pythonFolder",
                         type=str,
                         default=os.getcwd(),
                         help="The folder where the cocotb test file is.")

  argparser.add_argument("--testcase",
                         choices=['singleOut', 'multipleOut'],
                         default="singleOut",
                         help="Name of the testcase to run")

  argparser.add_argument(
      "sources",
      nargs="+",
      help="The list of verilog source files to be included.")

  return argparser.parse_args(args[1:])


class _IVerilogHandler:
  """ Class for handling icarus-verilog specific commands and patching."""

  def __init__(self):
    # Ensure that iverilog is available in path and it is at least iverilog v11
    try:
      out = subprocess.check_output(["iverilog", "-V"])
    except subprocess.CalledProcessError:
      raise Exception("iverilog not found in path")

    # find the 'Icarus Verilog version #' string and extract the version number
    # using a regex
    ver_re = r"Icarus Verilog version (\d+\.\d+)"
    ver_match = re.search(ver_re, out.decode("utf-8"))
    if ver_match is None:
      raise Exception("Could not find Icarus Verilog version")
    ver = ver_match.group(1)
    if float(ver) < 11:
      raise Exception(f"Icarus Verilog version must be >= 11, got {ver}")

  def extra_compile_args(self, objDir):
    # If no timescale is defined in the source code, icarus assumes a
    # timescale of '1'. This prevents cocotb from creating small timescale clocks.
    # Since a timescale is not emitted by default from export-verilog, make our
    # lives easier and create a minimum timescale through the command-line.
    cmd_file = os.path.join(objDir, "cmds.f")
    with open(cmd_file, "w+") as f:
      f.write("+timescale+1ns/1ps")

    return [f"-f{cmd_file}"]


def main():
  args = parseArgs(sys.argv)
  sources = [os.path.abspath(s) for s in args.sources]
  args.sources = sources

  if args.objdir is not None:
    objDir = args.objdir
  else:
    objDir = f"{os.path.basename(args.sources[0])}.d"
  objDir = os.path.abspath(objDir)
  if not os.path.exists(objDir):
    os.mkdir(objDir)
  os.chdir(objDir)

  # Ensure that system has 'make' available:
  try:
    subprocess.check_output(["make", "-v"])
  except subprocess.CalledProcessError:
    raise Exception(
        "'make' is not available, and is required to run cocotb tests.")

  try:
    if args.simulator == "icarus":
      simhandler = _IVerilogHandler()
    else:
      raise Exception(f"Unknown simulator: {simulator}")
  except Exception as e:
    raise Exception(f"Failed to initialize simulator handler: {e}")

  # Simulator-specific extra compile args.
  compileArgs = []
  if simhandler:
    compileArgs += simhandler.extra_compile_args(objDir)

  testmodule = "test_" + args.topLevel
  run(simulator=args.simulator,
      module=args.pythonModule,
      toplevel=args.topLevel,
      toplevel_lang="verilog",
      verilog_sources=sources,
      python_search=[args.pythonFolder],
      work_dir=objDir,
      testcase=args.testcase,
      compile_args=compileArgs)


if __name__ == "__main__":
  main()
