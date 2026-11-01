// Fault fixture: the loader must retain a DLL whose initialization reports failure.
static int attempts = 0;
extern "C" __declspec(dllexport) int initBalancedPlugin() { ++attempts; return 0; }
extern "C" __declspec(dllexport) int initBalancedBlocksPlugin() { ++attempts; return 0; }
extern "C" __declspec(dllexport) int aji_test_registration_attempts() { return attempts; }
