import * as su from '../util/SerialUtils.js'
import * as util from '../util/util.js'
import { GraphContext, GraphMem, Operation } from '../graph/GraphContext.js';

class BasePtr extends util.BaseAsyncObj{
  constructor(context, id=0, cb = null){
    super(context,cb)
    this.allocated = false;
    this.id = id;
    this.alloccb = this.alloccb.bind(this)
  }
  alloccb(err, id){
    this.id = id
    this.allocated = err == 0
    this.settle(this)
  }
  free(cb = null){
    this.when(()=>{
      this.ctx.free(this.id, cb)
      this.allocated = false
    })
  }
  get(N, offset,cb){
    this.when(()=>{
      const buf = this.ctx.makeCommand(6,(err,res)=>{
        if(err) return;
        this.ctx.gatherNext(res, cb)
      }, this.id, N, offset)
      this.ctx.sendBuffer(1, buf)
    })
  }
  set(data, offset=0, cb=null){
    this.when(()=>{
      const buf = this.ctx.makeCommand(7,(err,res)=>{
        if(err) console.error("buffer upload failed")
        cb?.()
      }, this.id, data.byteLength, offset)
      this.ctx.sendBuffer(1,util.b_cc(buf, data))
    })
  }
}

class CudaFunc extends util.BaseAsyncObj{
  static widths = {
    'ptr': 8, //ptr in this case is not an actual pointer but an id
    'rawptr': 8,
    'f64':8, 'f32':4, 'f16':2,
    'i64':8, 'i32':4, 'i16':2, 'i8':1,
    'u64':8, 'u32':4, 'u16':2, 'u8':1,
  }
  static argfuncs = {
    'ptr': 'Custom', //ptr in this case is not an actual pointer but an id
    'rawptr': 'setUint64',
    'f64':'setFloat64', 'f32':'setFloat32', 'f16':'Not permitted',
    'i64':'setInt64', 'i32':'setInt32', 'i16':'setInt16', 'i8':'setInt8',
    'u64':'setBigUint64', 'u32':'setUint32', 'u16':'setUint16', 'u8':'setUint8',
  }
  constructor(module, cb=null){
    super(module.ctx, cb)
    this.footprint = ['ptr', 'f32', 'u64']
    this.ptridxs = this.footprint.map((x,i)=>i).filter((i)=>this.footprint[i]=='ptr')
    this.width = this.footprint.map(x=>CudaFunc.widths[x]+2).reduce((a,b)=>a+b);
    this.createcb = this.createcb.bind(this)
  }

  createcb(err, id){
    if(err) throw Error("Error creating function")
    this.id = id
    this.settle(this)
  }

  call(...args){
    let argbuf = new Uint8Array(this.width)
    let v = new DataView(argbuf.buffer)
    let offset = 0;
    let depptrs = [];
    let defer = this.unrest;
    let graphContext = null;
    for(let i=0; i<this.ptridxs.length && !defer; i++){
      let idx = this.ptridxs[i];
      if(args[idx].ctx instanceof GraphContext){
        graphContext = graphContext??args[idx].graphContext;
        if(graphContext != args[idx].ctx) throw Error("Multiple graph contexts in one func call!")
      }
      defer |= this.footprint[idx]=='ptr' && args[idx].unrest;
    }
    const cb = graphContext? null:new util.Allcb(()=>{
      let buf = this.ctx.makeCommand(5, null, this.id, this.footprint.length, 0)
      this.ctx.sendBuffer(1,util.b_cc(buf, argbuf))
      if(depptrs.length>0) depptrs.forEach(ptr=>ptr.settle())
    })

    for(let i=0; i<this.footprint.length; i++){
      const argw = CudaFunc.widths[this.footprint[i]]
      if(argw){
        v.setUint16(offset, argw, true)
        offset += 2
        if(this.footprint[i] == 'ptr'){
          v.setUint16(offset-2, 0, true)
          if(!(args[i] instanceof BasePtr) && !(args[i] instanceof GraphMem)){
            throw Error("Argument "+i+ " should be pointer")
          }
          //v.setUint32(curoffset, ptr.id, true)
          /*if(defer){
            if(args[i].unrest){
              const curoffset = offset;
              if(!graphContext) cb.ex()
              args[i].when((ptr)=>{
                v.setUint32(curoffset, ptr.id, true)
                if(!graphContext && cb.c()){ //the good ol' wait-queue switcheroo (rhymes verbally)
                  ptr.unsettle() //lmao this really is \Delta type programming
                  depptrs.push(ptr);
                };
              })
            } else {
              v.setUint32(offset, args[i].id, true)
              if(!graphContext){
                args[i].unsettle()
                depptrs.push(ptr)
              }
            }
          } else {
            v.setUint32(offset, ptr.id, true)
          }*/
          const ptr = args[i]
          //greatly simplified via non async ids (why did I ever)
          console.log(ptr.id)
          v.setUint32(offset, ptr.id, true)
          if(defer && !graphContext){
            if(ptr.unrest){
              cb.ex()
              ptr.when(()=>{
                if(cb.c()){ //read the docs. no race condition.
                  depptrs.push(ptr)
                  ptr.unsettle()
                }
              })
            } else {
              depptrs.push(ptr)
              ptr.unsettle()
            }
          }
        } else{
          v[CudaFunc.argfuncs[this.footprint[i]]](offset,args[i],true)
        }
        offset += argw
      } else {
        throw Error("not yet implemented")
      }
    }
    if(graphContext){
      if(this.unrest) graphContext.externFuncs.add(this);
      let op = graphContext.cop(3, util.b_cc(argbuf), null, true);
      for(let i=0; i<this.ptridxs.length; i++){
        if(args[this.ptridxs[i]] instanceof BasePtr) graphContext.externPtrs.add(args[this.ptridxs[i]]);
      }
    } else {
      this.when(cb.c)
    }
  }
}

class CudaModule extends util.BaseAsyncObj{
  constructor(context, cb = null){
    super(context, cb);
    this.active = false;
    this.createcb = this.createcb.bind(this)
  }
  createcb(err, id){
    this.id = id
    this.active = err == 0
    this.settle()
  }
  createFunc(name, cb=null){
    let func = new CudaFunc(this, cb)
    this.when(()=>{
      const buf = this.ctx.makeCommand(4, func.createcb, this.id,name.length,0);
      this.ctx.sendBuffer(1,util.b_cc(buf, new TextEncoder().encode(name)))
    })
    return func
  }
}

export class PTXFile extends su.File{
  constructor(filecontext, path, cb){
    super(filecontext, path, cb)
  }
  compile=(cudacontext, cb=null)=>{
    let m = new CudaModule(cudacontext, cb)
    this.when(()=>{
      cudacontext.createModule(
        new TextEncoder().encode(new TextDecoder().decode(this.content)), 
        null, m
      )
    })
    return m
  }
}

export class BaseCudaContext{
  static version = 1;
  static baseResponseSize = 16 //Long+2*int

  constructor(cudaprocess){
    this.p = cudaprocess;
    this.p.seterrcb((data) => {
      console.error(`stderr: ${new TextDecoder().decode(data)}`);
    });
    this.p.setclosecb((code) => {
      console.log(`child process closed:\n ${new TextDecoder().decode(code)}`);
      this.p.cleanup() //implement this with proper handling of dangling requests
    }); 
    this.commandCounter = 1;
    this.ptrCounter = 1;
    this.fcmds={}

    this.stdout = new util.StreamCache()
    this.p.setoutcb(this.stdout.enqueue);
    this.handleResponse = this.handleResponse.bind(this)
    this.stdout.read(BaseCudaContext.baseResponseSize, this.handleResponse)
  }
  kill(){
    this.sendBuffer(8, new Uint8Array(0))
  }
  handleResponse(data){
    const v = new DataView(new Uint8Array(data).buffer);
    const cid = v.getBigUint64(0,true);
    if(this.fcmds[cid]===undefined) return
    this.fcmds[cid]?.(v.getUint32(8,true),v.getUint32(12,true))
    delete this.fcmds[cid]
    this.stdout.read(BaseCudaContext.baseResponseSize, this.handleResponse)
  }
  gatherNext(size, cb){
    this.stdout.read(size, cb)
  }
  makeHeader(info, bytes){
    let h = new Uint8Array(16);
    let v = new DataView(h.buffer);
    v.setUint32(0, BaseCudaContext.version, true);
    v.setUint32(4,info,true);
    v.setBigInt64(8,BigInt(bytes),true);
    return h
  }
  makeCommand(command, cb, arg1=0, arg2=0, arg3=0){
    const cid = this.getCommandId(cb);
    let h = new Uint8Array(24);
    let v = new DataView(h.buffer);
    v.setBigUint64(0, BigInt(cid), true);
    v.setUint32(8,command,true);
    v.setUint32(12,arg1,true);
    v.setUint32(16,arg2,true);
    v.setUint32(20,arg3,true);
    return h
  }
  sendBuffer(info, buffer){
    this.p.send(util.b_cc(this.makeHeader(info, buffer.byteLength), buffer))
  }
  malloc(size,cb=null){
    const ptrid = this.getPtrId()
    const ptr = new BasePtr(this, ptrid, cb)
    const buf = this.makeCommand(0, ptr.alloccb, 
      ptrid, size%4294967296, Math.floor(size/4294967296)
    ); //cursed buffer alignment
    this.sendBuffer(1,buf)
    return ptr;
  }
  free(ref, cb=null){
    const addr = ref.addr??ref;
    const buf = this.makeCommand(1, cb, addr,0,0);
    this.sendBuffer(1,buf)
  }
  createModule(code, cb=null, moduleref = null){
    const module = moduleref??new CudaModule(this, cb)
    const buf = this.makeCommand(2, module.createcb, code.byteLength,0,0);
    this.sendBuffer(1,util.b_cc(buf, code))
    return module;
  }
  //We use getters instead of direct incrementation for external purposes
  //(we are not the java-type people, I swear!)
  getCommandId(cb){ 
    this.fcmds[++this.commandCounter] = cb;
    return this.commandCounter
  }
  getPtrId(){
    return ++this.ptrCounter
  }
}