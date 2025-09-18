using Aqua, JET

Aqua.test_all(HybridDynamicModels; ambiguities=false, deps_compat=(check_extras = false))
JET.test_package(HybridDynamicModels)