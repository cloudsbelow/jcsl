export function b_cc(...bufs){
  let offsets = [];
  let coff = 0
  for(let i=0; i<bufs.length; i++){
    offsets.push(coff);
    coff+=bufs[i].byteLength
  }
  let res = new Uint8Array(coff)
  for(let i=0; i<bufs.length; i++){
    res.set(new Uint8Array(bufs[i].buffer),offsets[i])
  }
  return res
}

export class Allcb{
  constructor(cb, count = 1){
    this.cb=cb
    this.count=count
    this.c = this.c.bind(this)
    this.ex = this.ex.bind(this)
  }
  c=function(){
    this.count--
    if(this.count==0) this.cb();
    return this.count;
  }
  ex=function(){
    this.count++
    return this.c
  }
  rem=function(){
    return this.count
  }
}

export class Queue{
  constructor(){
    this.first = null;
    this.last = null;
  }
  enqueue(item){
    const link = {
      n:null,
      d:item,
    }
    if(this.first == null) this.first = link;
    else this.last.n = link;
    this.last = link;
  }
  peak(){
    return this.first;
  }
  dequeue(){
    const link = this.first;
    if(!link) return null;
    this.first = link.n
    return link.d;
  }
}

export class BaseAsyncObj{
  constructor(context, cb = null){
    if(cb && typeof cb != 'function') throw new Error("pass nonfunction");
    this.onSettle = new Queue()
    if(cb) this.onSettle.enqueue(cb)
    this.ctx = context
    this.unrest = 1; //by default, objects need one settle
    this.settle = this.settle.bind(this)
  }
  settle(){
    this.unrest--;
    while(this.onSettle.peak()){
      if(this.unrest) break;
      this.onSettle.dequeue()(this);
    }
  }
  unsettle(){
    this.unrest++
  }
  when(fn){
    if(this.unrest == 0){
      fn(this)
    } else {
      if(typeof fn != 'function') throw new Error("pass nonfunction");
      this.onSettle.enqueue(fn)
    }
    return this;
  }
}

export class StreamCache{
  constructor(){
    this.first = null
    this.last = null
    this.enqueued = 0
    this.waitfirst = null
    this.waitlast = null
    this.enqueue = this.enqueue.bind(this)
  }
  isReadable(num){
    return this.enqueued>=num
  }
  enqueue(buf){
    const link = {
      data: buf,
      next: null
    }
    if(this.first == null) this.first = link
    else this.last.next = link
    this.last = link
    this.enqueued += buf.byteLength
    while(this.enqueued>=this.waitfirst?.size){
      this.waitfirst.cb(this.readnow(this.waitfirst.size))
      this.waitfirst = this.waitfirst.next
    }
    return this.enqueued
  }
  readnow(size){
    if(!this.isReadable(size)) return null;
    if(size == 0) size = this.enqueued
    let c = new Uint8Array(size)
    let offset = 0;
    while(size-offset && size-offset>=this.first.data.byteLength){
      c.set(this.first.data, offset)
      offset += this.first.data.byteLength
      this.first = this.first.next
    }
    if(size-offset>0){
      c.set(this.first.data.subarray(0,size-offset))
      this.first.data = this.first.data.subarray(size-offset)
    }
    this.enqueued -= size
    return c
  }
  read(size,cb){
    if(size<=this.enqueued && this.waitfirst == null && this.enqueued!=0){
      cb(this.readnow(size))
    } 
    else {
      const link = {
        cb:cb,
        size:size,
        next:null,
      }
      if(this.waitfirst == null) this.waitfirst = link
      else this.waitlast.next = link
      this.waitlast = link
    }
  }
  empty(){
    return this.enqueued == 0;
  }
}
