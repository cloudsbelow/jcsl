


//const fake = new jcsl.FProcess('r/'+Date.now())
const real = new jcsl.RProcessClient('r/'+Date.now())
const c = new jcsl.BaseCudaContext(real)
window.addEventListener('beforeunload', ()=>{c.kill()})
const files = new jcsl.su.FileContextRemote('f')

const file = new jcsl.PTXFile(files, 'test.ptx')
const m = file.compile(c, ()=>{console.log("ok")})
const f = m.createFunc("_Z4fillPffy");
const a = c.malloc(256)
f.call(a, 1, BigInt(10));
a.set(new Float32Array([2,2,3,4]))
a.get(32, 0,(data)=>{
  console.log(new Float32Array(data.buffer))
})
