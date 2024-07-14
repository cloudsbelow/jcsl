import * as util from "../util.js"

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
  constructor(baseurl){
    this.base = baseurl+"/"
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

export class FileContextLocal extends util.BaseAsyncObj{
  constructor(basepath){
    this.base = basepath;
    this.fs = undefined;
    import('fs').then((fs)=>{this.settle(); this.fs=fs})
  }
  getfile=(file,cb)=>{
    const filepath = this.base+file;
    this.when(()=>{
      this.fs.readFile(filepath, (err, data)=>{
        if(err) return cb(err, null);
        cb(0,data)
      })
    });
  }
}

export class FileContextServer extends util.BaseAsyncObj{
  constructor(inurl, ctx){
    super(ctx)
    this.url = inurl;
  }
  getfile=(file, cb)=>{
    this.ctx.getfile(file, cb);
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

