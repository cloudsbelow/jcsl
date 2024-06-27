import * as util from "../util/util.js"

const commands = {
  empty:0,
  alloc:1,
  free:2,
  call:3,
  load:4,
}

export class Operation{
  constructor(idx, type, info, deps){
    this.idx = idx;
    this.type = type
    this.info = info
    this.deps = deps??[]
    this.offset = 0
  }
  adddep = dep=>this.deps.push(dep.idx);
  toBuf = (offset)=>{
    this.offset = offset;
    const header = new Uint32Array(2+this.deps.length)
    header[0] = this.type;
    header[1] = this.deps.length;
    for(let i=0; i<this.deps.length; i++) header[i+2] = this.deps[i];
    return util.b_cc(header, this.info);
  }
}

export class GraphMem{ 
  //This object is a lot like a basePtr but it is not identical
  //this enforces graph order operations THE OTHER DOES NOT (!Important!)
  constructor(context, id, size, deps = null){
    this.ctx = context;
    this.ops = [new Operation("createPlaceholder", deps, true)]
    this.lastmodidx = 0;
    this.freed = false;
    this.id = id;
    this.size = size;
  }
  addop=(op, mod = true)=>{
    if(!mod){
      op.adddep(this.ops[this.lastmodidx])
      this.ops.push(op)
    }
    if(mod){
      for(this.lastmodidx; this.lastmodidx<this.ops.length; this.lastmodidx++){
        op.adddep(this.ops[i])
      }
      this.ops.push(op)
    }
  }
  free=()=>{
    if(this.freed) return;
    let freeOp = this.ctx.cop(commands.free, new Uint32Array([this.id]), null, true)
    this.addop(freeOp)
    this.freed = true
  }
}

export class GraphContext{
  constructor(parent){
    this.base = parent instanceof GraphContext? parent.base:parent;
    this.gid = objs.globalGC? ++objs.globalGC : (objs.globalGC=1);
    this.allocs = [];
    this.ops = [];
    this.externFuncs = new Set();
    this.externPtrs = new Set();
    this.parent = parent
    this.graphDescriptor()
  }
  cop(type, info, deps, mod = true){
    const op = new Operation(this.ops.length, type, info, deps)
    this.ops.push(op)
    return op
  }
  malloc=(size)=>{
    const mem = new GraphMem(this, this.parent.getPtrId(), size)
    this.allocs.push(mem)
  }
  compile=()=>{
    this.allocs.forEach(mem=>mem.free())
    offset = 16;
    const comps = this.ops.map(op=>{
      const buf = op.toBuf(offset)
      offset += buf.byteLength
      return buf
    })
    this.graphDescriptor = new Uint8Array(offset);
    const v = new DataView(this.graphDescriptor.buffer)
    v.setUint32(8,this.ops.length, true);
    comps.forEach((buf, i)=>this.graphDescriptor.set(buf, this.ops[i].offset))
    console.log(this.graphDescriptor)
    return this
  }
  
}