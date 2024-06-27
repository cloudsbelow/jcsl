import * as util from './util.js'

export class CProcess{
  constructor(path, cb){
    this.p = undefined
    //require('child_process').spawn(path)
    import('child_process').then((cp)=>{
      this.p = cp.spawn(path)
      cb()
    })
  }
  setoutcb(cb){
    this.p.stdout.on('data', (data)=>{
      //console.log(data)
      cb(data)
    });
  }
  seterrcb(cb){
    this.p.stderr.on('data', cb);
  }
  setclosecb(cb){
    this.p.on('close', cb);
  }
  send(data){
    //for(let i=0; i<data.length; i++){
      //console.log(data[i])
    //}
    //console.log(data)
    this.p.stdin.write(data);
  }
}

export class RProcessClient extends util.BaseAsyncObj{
  constructor(url){ //This would be a (really) good place to use sockets BUT I DUNNO HOW SO POLLING HERE WE GO
    super(null, )
    this.url=url;
    const xhr = new XMLHttpRequest();
    xhr.onreadystatechange = (stat)=>{
      //console.log(stat, this)
      if(xhr.status === 200 && xhr.readyState === XMLHttpRequest.DONE){
        console.log('established connection')
        this.settle();
      }
    }
    xhr.open('GET',this.url+"/open")
    xhr.send()
    //console.log("h")
  }
  poll=(end, cb)=>{
    const xhr = new XMLHttpRequest();
    xhr.timeout = 0;
    xhr.responseType = "arraybuffer"
    xhr.onreadystatechange = ()=>{
      if(xhr.readyState === XMLHttpRequest.DONE){
        if(xhr.status === 200) {
          //console.log(new Uint8Array(xhr.response))
          cb(new Uint8Array(xhr.response));
          this.poll(end, cb)
        } else console.error("huh")
      }
    }
    xhr.open('GET',this.url+end)
    xhr.send()
  }
  setoutcb=(cb)=>{
    this.when(()=>this.poll('/out', cb))
  }
  seterrcb=(cb)=>{
    this.when(()=>this.poll('/err',cb))
  }
  setclosecb=(cb)=>{
    this.when(()=>this.poll('/close',cb))
  }
  send(data){
    const xhr = new XMLHttpRequest();
    xhr.onreadystatechange = ()=>{
      if(xhr.readyState === XMLHttpRequest.DONE){
        if(xhr.status === 200) {
        } else console.error('idk')
      }
    };
    xhr.open("PUT", this.url+"/in");
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    this.when(()=>xhr.send(data));
  }
}

export class RProcessServer{ //not the whole server, just a part
  constructor(id, path, onopen, onclose){
    this.id = id //eventually reshape this to per-client stuff ig
    this.p = new CProcess(path, ()=>{
      onopen()
      this.p.setoutcb(this.out.enqueue)
      this.p.seterrcb(this.err.enqueue)
    })
    this.out = new util.StreamCache()
    this.err = new util.StreamCache()
    this.onclose = onclose
  }
  processRequest = (req, res)=>{
    const spath = req.url.split('/');
    if(spath[2]!=this.id) return false //as said above
    switch(spath[3]){
      case 'out':
        this.out.read(0,(data)=>{
          res.writeHead(200, {'Content-Type': 'application/octet-stream'});
          res.write(data);
          res.end()
        })
        break;
      case 'err':
        this.err.read(0, (data)=>{
          res.writeHead(200, {'Content-Type': 'application/octet-stream'});
          res.write(data)
          res.end()
        })
        break
      case 'close':
        this.p.setclosecb((data)=>{
          res.writeHead(200, {'Content-Type': 'application/octet-stream'})
          res.write('Program closed with code '+data)
          res.end()
          console.log("The process with id "+this.id+" has closed");
          this.onclose(this)
        })
        break
      case 'in':
        req.on('data',(data)=>{
          this.p.send(new Uint8Array(data))  
        })
        //req.pipe(this.p.p.stdin, {end:false})
        req.on('end', ()=>{
          res.writeHead(200);
          res.end();
        })
        break
      default:
        console.error("no ", spath[3]);

    }
  }
}

export class File extends util.BaseAsyncObj{
  constructor(context, path, cb){
    super(context, cb)
    this.found = false;
    context.getfile(path, this.foundcb)
  }
  foundcb = (err, content)=>{
    this.content = content
    this.found = err == 0;
    this.settle()
  }
}

export class FileContextRemote{
  constructor(basepath){
    this.base = basepath+"/"
  }
  getfile=(file, cb)=>{
    const xhr = new XMLHttpRequest();
    xhr.timeout = 0;
    xhr.responseType = "arraybuffer";
    xhr.onreadystatechange = ()=>{
      if(xhr.readyState === XMLHttpRequest.DONE){
        if(xhr.status === 200) {
          cb(0, xhr.response);
        } else cb(xhr.status, null)
      }
    }
    xhr.open('GET',this.base+file)
    xhr.send() 
  }
}

export class FileContextServer extends util.BaseAsyncObj{
  constructor(url, basepath){
    super()
    this.url = url;
    this.base = basepath;
    this.fs = undefined;
    import('fs').then((fs)=>{this.settle(); this.fs=fs})
  }
  getfile=(file, cb)=>{
    const filepath = this.base+file;
    this.when(()=>{
      this.fs.readFile(filepath, (err, data)=>{
        if(err) return cb(err, null);
        cb(0,data)
      })
    });
  }
  processRequest=(req, res)=>{
    const spath = req.url.split('/');
    if(spath[1]!=this.url) return false
    this.getfile(req.url.slice(1+this.base.length),(err, data)=>{
      if(err){
        res.writeHead(err, { 'Content-Type': 'text/plain' });
        return res.end()
      }
      res.writeHead(200, {'Content-Type': 'application/octet-stream'})
      res.write(data)
      res.end()
    })
  }
}

