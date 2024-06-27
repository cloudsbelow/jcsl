



const c = new jcsl.BaseCudaContext(new jcsl.su.RProcessClient('r/'+Date.now()))
window.addEventListener('beforeunload', c.kill)
const files = new jcsl.su.FileContextRemote('f')
const file = new jcsl.PTXFile(files, 'test.ptx')
const m = file.compile(c, ()=>{console.log("ok")})
const f = m.createFunc("_Z4fillPffy");
const a = c.malloc(256)
f.call(a, 1, BigInt(10));
a.set(new Float32Array([1,2,3,4]))
a.get(32, 0,(data)=>{
  console.log(new Float32Array(data.buffer))
})
