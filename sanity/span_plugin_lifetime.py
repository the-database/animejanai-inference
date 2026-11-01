"""Run named_profiles in-process, then check the failed registration DLL is retained."""
import ctypes as C
import os
from pathlib import Path
import runpy
import sys

script = Path(__file__).with_name('named_profiles.py')
sys.argv[0] = str(script)
state = runpy.run_path(str(script), run_name='__main__')
plugin = os.environ['AJI_TEST_FAILED_PLUGIN']
get_module = C.WinDLL('kernel32').GetModuleHandleW
get_module.argtypes = [C.c_wchar_p]
get_module.restype = C.c_void_p
assert get_module(plugin), 'Failed registration unloaded the creator DLL'
probe = C.CDLL(plugin).aji_test_registration_attempts
probe.restype = C.c_int
assert probe() == 1, 'Failed initialization was retried during profile switching'
print('PASS: failed registration DLL retained; initialization attempted once')
