"""Exercise named-model building, cache reuse and profile changes in one process.

Requires Windows, NumPy and the backend's CUDA 13 runtime. Run without concurrent
GPU work. Inputs are raw NV12 clips at --width/--height.
The output directory records frames and timings; use a new directory for each run.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--backend',type=Path,required=True)
p.add_argument('--runtime',type=Path,required=True)
p.add_argument('--conf',type=Path,required=True)
p.add_argument('--models',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--frames',type=Path,action='append',required=True)
p.add_argument('--slots',default='1,2,3,4')
p.add_argument('--width',type=int,default=1920)
p.add_argument('--height',type=int,default=1080)
p.add_argument('--clean-cache',action='store_true')
p.add_argument('--blocking',action='store_true',help='Build synchronously instead of polling the async builder')
a=p.parse_args()
slots=[int(s) for s in a.slots.split(',')]
a.output.mkdir(parents=True,exist_ok=False)
initial=list(a.models.glob('*.engine'))
if a.clean_cache and initial:
    raise RuntimeError('The clean-cache check requires a model directory without engines')
cookies=[os.add_dll_directory(str(d.resolve())) for d in {a.runtime,a.backend.parent}]
os.environ['PATH']=str(a.runtime.resolve())+os.pathsep+os.environ['PATH']
cuda=C.CDLL(str((a.runtime/'cudart64_13.dll').resolve()))
aji=C.CDLL(str(a.backend.resolve()))
class Params(C.Structure):
    _fields_=[('api',C.c_uint32),('context',C.c_void_p),('conf',C.c_char_p),('models',C.c_char_p),('trtexec',C.c_char_p),('env',C.c_char_p),('slot',C.c_int),('rife',C.c_char_p),('async_build',C.c_int),('d3d',C.c_void_p),('engine',C.c_char_p),('maxw',C.c_int),('maxh',C.c_int),('log',C.c_void_p),('opaque',C.c_void_p)]
class Frame(C.Structure):
    _fields_=[('width',C.c_int),('height',C.c_int),('format',C.c_int),('matrix',C.c_int),('range',C.c_int),('siting',C.c_int),('plane',C.c_void_p*3),('stride',C.c_ssize_t*3)]
def bind(lib,name,args,result=C.c_int):
    fn=getattr(lib,name);fn.argtypes=args;fn.restype=result;return fn
malloc=bind(cuda,'cudaMalloc',[C.POINTER(C.c_void_p),C.c_size_t])
copy=bind(cuda,'cudaMemcpy',[C.c_void_p,C.c_void_p,C.c_size_t,C.c_int])
copy_async=bind(cuda,'cudaMemcpyAsync',[C.c_void_p,C.c_void_p,C.c_size_t,C.c_int,C.c_void_p])
sync=bind(cuda,'cudaStreamSynchronize',[C.c_void_p])
free=bind(cuda,'cudaFree',[C.c_void_p])
create_stream=bind(cuda,'cudaStreamCreateWithFlags',[C.POINTER(C.c_void_p),C.c_uint])
destroy_stream=bind(cuda,'cudaStreamDestroy',[C.c_void_p])
create=bind(aji,'aji_create',[C.POINTER(Params)],C.c_void_p)
configure=bind(aji,'aji_configure',[C.c_void_p,C.c_int,C.c_int,C.c_double,C.POINTER(C.c_int),C.POINTER(C.c_int)])
set_slot=bind(aji,'aji_set_slot',[C.c_void_p,C.c_int])
poll=bind(aji,'aji_poll',[C.c_void_p])
infer=bind(aji,'aji_infer',[C.c_void_p,C.POINTER(Frame),C.POINTER(Frame),C.c_void_p])
error=bind(aji,'aji_last_error',[C.c_void_p],C.c_char_p)
current_log=bind(aji,'aji_current_log',[C.c_void_p],C.c_char_p)
destroy=bind(aji,'aji_destroy',[C.POINTER(C.c_void_p)],None)
messages=[]
@C.CFUNCTYPE(None,C.c_void_p,C.c_int,C.c_char_p)
def logger(_,level,message):
    messages.append(message.decode('utf-8','replace'))
def ck(rc):
    if rc: raise RuntimeError(f'CUDA error {rc}')
def ajck(rc):
    if rc<0: raise RuntimeError(error(ctx))
params=Params(api=9,conf=os.fsencode(a.conf.resolve()),models=os.fsencode(a.models.resolve()),
    trtexec=os.fsencode((a.runtime/'trtexec.exe').resolve()),slot=slots[0],async_build=0 if a.blocking else 1,
    log=C.cast(logger,C.c_void_p))
ctx=C.c_void_p(create(C.byref(params)))
if not ctx: raise RuntimeError('aji_create failed')
stream=C.c_void_p();ck(create_stream(C.byref(stream),1))
w,h=a.width,a.height;size=w*h*3//2
inputs=[]
for file in a.frames:
    clip=np.fromfile(file,dtype=np.uint8).reshape(-1,size)
    inputs.extend(clip[i].copy() for i in range(0,min(len(clip),24),6))
ip,op=C.c_void_p(),C.c_void_p();ck(malloc(C.byref(ip),size))
plane=w*2*h*2*2;ck(malloc(C.byref(op),plane*3))
inp=Frame(w,h,1,2,1,1,(C.c_void_p*3)(ip.value,ip.value+w*h,None),(C.c_ssize_t*3)(w,w,0))
out=Frame(w*2,h*2,3,2,1,1,(C.c_void_p*3)(op.value,op.value+plane,op.value+plane*2),(C.c_ssize_t*3)(w*4,w*4,w*4))
host=np.empty(plane*3,dtype=np.uint8)
rows=[];first={};timings=[]
def setup(slot):
    ck(sync(stream));ajck(set_slot(ctx,slot))
    start=time.monotonic();ow,oh=C.c_int(),C.c_int()
    rc=configure(ctx,w,h,24000/1001,C.byref(ow),C.byref(oh));ajck(rc)
    while rc==0:
        if time.monotonic()-start>1800: raise TimeoutError('Engine build did not complete')
        ready=poll(ctx);ajck(ready)
        if ready:
            rc=configure(ctx,w,h,24000/1001,C.byref(ow),C.byref(oh));ajck(rc)
        else: time.sleep(.25)
    if (ow.value,oh.value)!=(w*2,h*2): raise RuntimeError('Expected 2x output')
    timings.append(dict(slot=slot,seconds=time.monotonic()-start,
                        configuration=(current_log(ctx) or b'').decode('utf-8','replace')))
for slot in slots:
    print('Configuring slot',slot,flush=True)
    setup(slot)
    for i,data in enumerate(inputs):
        # Order upload and inference on the same nonblocking stream. A default-
        # stream pageable H2D copy can return before the device transfer finishes.
        ck(copy_async(ip,C.c_void_p(data.ctypes.data),size,1,stream))
        ajck(infer(ctx,C.byref(inp),C.byref(out),stream));ck(sync(stream))
        ck(copy(C.c_void_p(host.ctypes.data),op,host.nbytes,2))
        digest=hashlib.sha256(host).hexdigest()
        host.tofile(a.output/f'slot-{slot}-frame-{i}.yuv444p16')
        rows.append(dict(slot=slot,frame=i,sha256=digest))
        if i==0:first[slot]=digest
    print('Completed slot',slot,'configure seconds',round(timings[-1]['seconds'],3),flush=True)
for slot in reversed(slots):
    setup(slot)
    ck(copy_async(ip,C.c_void_p(inputs[0].ctypes.data),size,1,stream))
    ajck(infer(ctx,C.byref(inp),C.byref(out),stream));ck(sync(stream))
    ck(copy(C.c_void_p(host.ctypes.data),op,host.nbytes,2))
    if hashlib.sha256(host).hexdigest()!=first[slot]: raise RuntimeError(f'Slot {slot} changed after switching')
ck(sync(stream));destroy(C.byref(ctx));ck(free(ip));ck(free(op));ck(destroy_stream(stream))
(a.output/'backend.log').write_text('\n'.join(messages))
result=dict(slots=slots,rows=rows,configure=timings,cache_initial=len(initial),
    engines=[p.name for p in a.models.glob('*.engine')],switching_identical=True,width=w,height=h)
(a.output/'results.json').write_text(json.dumps(result,indent=2))
print('PASS: named models/cache reuse/profile switching; frames',len(rows),flush=True)
