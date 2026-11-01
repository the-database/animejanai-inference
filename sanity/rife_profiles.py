"""Build/load stock and warp RIFE, compare real sequences, then reload stock.

Requires Windows, NumPy and the backend's CUDA 13 runtime. Use an isolated model
directory containing only the two RIFE exports. Inputs are raw 1920x1080 NV12
clips. Run without other GPU work; --clean-cache asserts an initially empty cache.
"""
import argparse,ctypes as C,hashlib,json,os,time
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--backend',type=Path,required=True)
p.add_argument('--runtime',type=Path,required=True)
p.add_argument('--models',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--frames',type=Path,action='append',required=True)
p.add_argument('--clean-cache',action='store_true')
p.add_argument('--width',type=int,default=1920)
p.add_argument('--height',type=int,default=1080)
p.add_argument('--blocking',action='store_true')
p.add_argument('--expect-fallback',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
if a.clean_cache and list(a.models.glob('*.engine')):raise RuntimeError('Expected an empty engine cache')
cookies=[os.add_dll_directory(str(d.resolve())) for d in {a.runtime,a.backend.parent}]
os.environ['PATH']=str(a.runtime.resolve())+os.pathsep+os.environ['PATH']
cuda=C.CDLL(str((a.runtime/'cudart64_13.dll').resolve()));aji=C.CDLL(str(a.backend.resolve()))
class Params(C.Structure):
    _fields_=[('api',C.c_uint32),('context',C.c_void_p),('conf',C.c_char_p),('models',C.c_char_p),('trtexec',C.c_char_p),('env',C.c_char_p),('slot',C.c_int),('rife',C.c_char_p),('async_build',C.c_int),('d3d',C.c_void_p),('engine',C.c_char_p),('maxw',C.c_int),('maxh',C.c_int),('log',C.c_void_p),('opaque',C.c_void_p)]
class Frame(C.Structure):
    _fields_=[('width',C.c_int),('height',C.c_int),('format',C.c_int),('matrix',C.c_int),('range',C.c_int),('siting',C.c_int),('plane',C.c_void_p*3),('stride',C.c_ssize_t*3)]
def bind(lib,name,args,result=C.c_int):
    f=getattr(lib,name);f.argtypes=args;f.restype=result;return f
malloc=bind(cuda,'cudaMalloc',[C.POINTER(C.c_void_p),C.c_size_t]);free=bind(cuda,'cudaFree',[C.c_void_p])
copy=bind(cuda,'cudaMemcpy',[C.c_void_p,C.c_void_p,C.c_size_t,C.c_int])
copy_async=bind(cuda,'cudaMemcpyAsync',[C.c_void_p,C.c_void_p,C.c_size_t,C.c_int,C.c_void_p])
make_stream=bind(cuda,'cudaStreamCreateWithFlags',[C.POINTER(C.c_void_p),C.c_uint])
sync=bind(cuda,'cudaStreamSynchronize',[C.c_void_p]);end_stream=bind(cuda,'cudaStreamDestroy',[C.c_void_p])
create=bind(aji,'aji_create',[C.POINTER(Params)],C.c_void_p)
configure=bind(aji,'aji_configure',[C.c_void_p,C.c_int,C.c_int,C.c_double,C.POINTER(C.c_int),C.POINTER(C.c_int)])
set_slot=bind(aji,'aji_set_slot',[C.c_void_p,C.c_int])
poll=bind(aji,'aji_poll',[C.c_void_p]);factor=bind(aji,'aji_rife_factor',[C.c_void_p,C.POINTER(C.c_int),C.POINTER(C.c_int)])
infer=bind(aji,'aji_infer_rife',[C.c_void_p,C.POINTER(Frame),C.POINTER(Frame),C.c_double,C.POINTER(Frame),C.c_void_p])
last_error=bind(aji,'aji_last_error',[C.c_void_p],C.c_char_p);destroy=bind(aji,'aji_destroy',[C.POINTER(C.c_void_p)],None)
current_log=bind(aji,'aji_current_log',[C.c_void_p],C.c_char_p)
messages=[]
@C.CFUNCTYPE(None,C.c_void_p,C.c_int,C.c_char_p)
def logger(_,level,msg):messages.append(msg.decode('utf-8','replace'))
def ck(rc):
    if rc:raise RuntimeError(f'CUDA {rc}')
def check(rc):
    if rc<0:raise RuntimeError(last_error(ctx))
conf=a.output/'rife.conf'
conf.write_text('[global]\nbackend=TensorRT\n[slot_1]\nchain_1_rife=yes\nchain_1_rife_model=422\nchain_1_rife_factor_numerator=2\nchain_1_rife_factor_denominator=1\nchain_1_rife_before_upscale=yes\nchain_1_rife_scene_detect_threshold=0.150\n')
stock_conf=conf.read_text()
conf.write_text(stock_conf+'\nchain_1_rife_fp16_warp=no\n[slot_2]'+stock_conf.split('[slot_1]')[1]+'\nchain_1_rife_fp16_warp=yes\n')
params=Params(api=9,conf=os.fsencode(conf.resolve()),models=os.fsencode(a.models.resolve()),
 trtexec=os.fsencode((a.runtime/'trtexec.exe').resolve()),slot=1,rife=os.fsencode(a.models.resolve()),
 async_build=0 if a.blocking else 1,log=C.cast(logger,C.c_void_p))
ctx=C.c_void_p(create(C.byref(params)))
if not ctx:raise RuntimeError('aji_create failed')
stream=C.c_void_p();ck(make_stream(C.byref(stream),1))
w,h=a.width,a.height
size=w*h*3//2;buffers=[];frames=[]
for _ in range(3):
 ptr=C.c_void_p();ck(malloc(C.byref(ptr),size));buffers.append(ptr)
 frames.append(Frame(w,h,1,2,1,1,(C.c_void_p*3)(ptr.value,ptr.value+w*h,None),(C.c_ssize_t*3)(w,w,0)))
clips=[np.fromfile(f,dtype=np.uint8).reshape(-1,size) for f in a.frames]
host=np.empty(size,dtype=np.uint8);rows=[];timings=[];references={};reset_reference={}
for mode in ['stock','warp','stock-reloaded']:
 check(set_slot(ctx,2 if mode=='warp' else 1))
 ck(sync(stream));start=time.monotonic();ow,oh=C.c_int(),C.c_int();num,den=C.c_int(),C.c_int()
 print('Configuring',mode,flush=True)
 check(configure(ctx,w,h,24000/1001,C.byref(ow),C.byref(oh)))
 while not factor(ctx,C.byref(num),C.byref(den)):
  if time.monotonic()-start>1800:raise TimeoutError('RIFE engine build timeout')
  ready=poll(ctx);check(ready)
  if ready:check(configure(ctx,w,h,24000/1001,C.byref(ow),C.byref(oh)))
  else:time.sleep(.25)
 if (num.value,den.value)!=(2,1):raise RuntimeError('Expected RIFE 2x')
 config=(current_log(ctx) or b'').decode('utf-8','replace')
 if mode=='warp':
  assert ('Using stock RIFE:' in config)==a.expect_fallback,config
 else:assert 'Using stock RIFE:' not in config,config
 timings.append(dict(mode=mode,seconds=time.monotonic()-start,configuration=config))
 pairs=[]
 for k,clip in enumerate(clips):
  pairs.extend((f'clip{k}-{i}',clip[i],clip[i+1],.5) for i in range(min(len(clip)-1,23)))
 for t in [.25,.75,.5]:pairs.append((f'timestep-{t}',clips[-1][8],clips[-1][9],t))
 black=np.full(size,128,dtype=np.uint8);black[:w*h]=16
 white=black.copy();white[:w*h]=235;pairs.append(('cut',black,white,.5))
 if mode=='stock-reloaded':pairs=pairs[:2]+pairs[-1:]
 for label,left,right,t in pairs:
  # Upload on the inference stream; default-stream pageable copies can still
  # have a device transfer in flight when a nonblocking stream starts inference.
  for i,data in enumerate([left,right]):ck(copy_async(buffers[i],C.c_void_p(data.ctypes.data),size,1,stream))
  rc=infer(ctx,C.byref(frames[0]),C.byref(frames[1]),t,C.byref(frames[2]),stream);check(rc);ck(sync(stream))
  row=dict(mode=mode,label=label,status=rc)
  if rc==0:
   ck(copy(C.c_void_p(host.ctypes.data),buffers[2],size,2));row['sha256']=hashlib.sha256(host).hexdigest()
   if mode=='stock':references[label]=host.copy()
   elif mode=='warp':
    delta=host.astype(np.float32)-references[label].astype(np.float32)
    row['mse']=float(np.mean(delta*delta));row['psnr_db']=float(10*np.log10(255**2/row['mse'])) if row['mse'] else 'infinite'
    if a.expect_fallback:assert row['mse']==0,'Fallback differs from stock'
  if mode=='stock':reset_reference[label]=row
  else:
   ref=reset_reference[label]
   if row['status']!=ref['status']:raise RuntimeError('Scene decision mismatch: '+label)
   if mode=='stock-reloaded' and row.get('sha256')!=ref.get('sha256'):raise RuntimeError('Stock changed after switching')
  rows.append(row)
 print('Completed',mode,'configure seconds',round(timings[-1]['seconds'],3),flush=True)
destroy(C.byref(ctx));ck(end_stream(stream))
for ptr in buffers:ck(free(ptr))
values=[r['mse'] for r in rows if 'mse' in r]
result=dict(rows=rows,configure=timings,warp_frames=len(values),
 aggregate_psnr_db=float(10*np.log10(255**2/np.mean(values))) if np.mean(values) else 'infinite',
 worst_psnr_db=min((r['psnr_db'] for r in rows if isinstance(r.get('psnr_db'),float)),default='infinite'),
 switching_identical=True,engines=[f.name for f in a.models.glob('*.engine')])
expected=1 if a.expect_fallback else 2
if len(list(a.models.glob('*.engine')))!=expected:raise RuntimeError(f'Expected {expected} engine cache entries')
(a.output/'results.json').write_text(json.dumps(result,indent=2));(a.output/'backend.log').write_text('\n'.join(messages))
print('PASS: RIFE build/load / distinct caches / switching',flush=True)
