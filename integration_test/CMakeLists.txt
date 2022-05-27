set(STANDALONE_INTEGRATION_TEST_DEPENDS
  FileCheck count not split-file
  standalone-opt
  circt-rtl-sim
  firtool)

set(STANDALONE_INTEGRATION_TIMEOUT 60) # Set a 60s timeout on individual tests.
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  INTEGRATION_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_lit_testsuite(check-standalone-integration "Running the thesis integration tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CIRCT_INTEGRATION_TEST_DEPENDS}
  )
set_target_properties(check-standalone-integration PROPERTIES FOLDER "IntegrationTests")

add_lit_testsuites(STANDALONE_INTEGRATION ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${STANDALONE_INTEGRATION_TEST_DEPS}
)
