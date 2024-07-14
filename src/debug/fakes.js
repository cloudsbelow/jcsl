export class FProcess{
  constructor(path, cb){
    this.out = null;
    this.err = null;
    this.close = null;
  }
  setoutcb(cb){
    this.out = cb
  }
  seterrcb(cb){
    this.err = cb
  }
  setclosecb(cb){
    this.close = cb
  }
  send(data){
    //for(let i=0; i<data.length; i++){
      //console.log(data[i])
    //}
    //console.log(data)
    console.log("Fake process gets: ",data);
  }
  respond(id, err, res){
    let packet = new Uint8Array(16)
    const v = new DataView(packet.buffer)
    v.setUint32(0, id, true)
    v.setUint32(8, err, true)
    v.setUint32(12, res, true)
    this.out(packet)
  }
}