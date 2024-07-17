import * as util from './../util.js'
import { RequestFactory } from './genericreq.js'

export class CProcess extends util.BaseAsyncObj{
  constructor(path, cb){
    super(null, cb)
    this.p = undefined
    //require('child_process').spawn(path)
    import('child_process').then((cp)=>{
      this.p = cp.spawn(path)
      //not race condition because settle process entirely sync
      //(and this object is never unsettled)
      //I feel as though leaving this through the gc would be
      //excessive for some reason I can't quite describe...
      //turns out this doesn't work now for some reason? cool.
      //this.send = this.p.stdin.write 
      this.settle()
    })
  }
  setoutcb(cb){
    this.when(()=>{
      this.p.stdout.on('data', (data)=>{
        //console.log(data)
        cb(data)
      });
    })
  }
  seterrcb(cb){
    this.when(()=>this.p.stderr.on('data', cb));
  }
  setclosecb(cb){
    this.when(()=>this.p.on('close', cb));
  }
  send(data){
    //for(let i=0; i<data.length; i++){
      //console.log(data[i])
    //}
    //console.log(data)
    this.when(()=>{
      this.p.stdin.write(data);
    })
  }
}

export class RProcessClient extends util.BaseAsyncObj{
  constructor(url,cb){ //This would be a (really) good place to use sockets BUT I DUNNO HOW SO POLLING HERE WE GO
    super(null, cb)
    this.url=url;
    this.connection = new RequestFactory(this.url)
    this.connection.send('/open', 'GET', ()=>{
      console.log("connection established: "+this.url)
      this.settle()
    })
    this.sendbuffer = new util.Flushable()
    this.flfn = (data)=>{
      this.connection.send('/in','PUT',()=>{
        this.sendbuffer.flush(flfn)
      },data)
    }
    this.sendbuffer.flush(flfn)
    /*const xhr = new XMLHttpRequest();
    xhr.onreadystatechange = (stat)=>{
      //console.log(stat, this)
      if(xhr.status === 200 && xhr.readyState === XMLHttpRequest.DONE){
        console.log('established connection')
        this.settle();
      }
    }
    xhr.open('GET',this.url+"/open")
    xhr.send()
    //console.log("h")*/
  }
  poll=(end, cb)=>{
    this.connection.send(end, 'GET', (data)=>{
      cb(data)
      this.poll(end, cb)
    })

    /*const xhr = new XMLHttpRequest();
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
    xhr.send()*/
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
    //this.connection.send('/in','PUT',()=>{},data)
    if(!(data instanceof Uint8Array)) throw Error();
    this.sendbuffer.enqueue(data);
    /*const xhr = new XMLHttpRequest();
    xhr.onreadystatechange = ()=>{
      if(xhr.readyState === XMLHttpRequest.DONE){
        if(xhr.status === 200) {
        } else console.error('idk')
      }
    };
    xhr.open("PUT", this.url+"/in");
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    this.when(()=>xhr.send(data));*/
  }
}

export class RProcessServer{ //not the whole server, just a part
  constructor(id, process, onopen, onclose){
    this.id = id //eventually reshape this to per-client stuff ig
    this.onclose = onclose
    this.p = process
    this.out = new util.StreamCache()
    this.err = new util.StreamCache()
    this.p.when(()=>{
      onopen()
    })
    this.p.setoutcb(this.out.enqueue)
    this.p.seterrcb(this.err.enqueue)
    /*new CProcess(path, ()=>{
      onopen()
      this.p.setoutcb(this.out.enqueue)
      this.p.seterrcb(this.err.enqueue)
    })*/
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