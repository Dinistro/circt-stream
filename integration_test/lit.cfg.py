# -*- Python -*-

import os
import platform
import re
import shutil
import subprocess
import tempfile
import warnings

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'Stream'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.td', '.mlir', '.ll', '.fir', '.sv', '.py', '.tcl']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.circt_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))
config.substitutions.append(('%shlibdir', config.circt_shlib_dir))
config.substitutions.append(('%INC%', config.circt_include_dir))
config.substitutions.append(
    ('%TCL_PATH%', config.circt_src_root + '/build/lib/Bindings/Tcl/'))
config.substitutions.append(('%PROJECT_SOURCE%', config.project_src_root))

llvm_config.with_system_environment(['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

llvm_config.use_default_substitutions()

# Set the timeout, if requested.
if config.timeout is not None and config.timeout != "":
  lit_config.maxIndividualTestTime = int(config.timeout)

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = [
    'Inputs', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt', 'lit.cfg.py',
    'lit.local.cfg.py'
]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.circt_obj_root, 'integration_test')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)
# Substitute '%l' with the path to the build lib dir.

tool_dirs = [
    config.circt_tools_dir, config.circt_utils_dir, config.mlir_tools_dir,
    config.llvm_tools_dir, config.project_tools_dir
]
tools = [
    'stream-opt',
    'firtool',
    'circt-rtl-sim.py',
]

# Enable python if its path was configured
if config.python_executable != "":
  tool_dirs.append(os.path.dirname(config.python_executable))
  config.available_features.add('python')
  config.substitutions.append(('%PYTHON%', config.python_executable))

# Enable Icarus Verilog as a fallback if no other ieee-sim was detected.
if config.iverilog_path != "":
  tool_dirs.append(os.path.dirname(config.iverilog_path))
  tools.append('iverilog')
  tools.append('vvp')
  config.available_features.add('iverilog')
  config.substitutions.append(('%iverilog', config.iverilog_path))

llvm_config.add_tool_substitutions(tools, tool_dirs)

# cocotb availability
try:
  import cocotb
  import cocotb_test
  config.available_features.add('cocotb')
except ImportError:
  pass
