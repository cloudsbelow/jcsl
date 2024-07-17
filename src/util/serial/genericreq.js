import * as util from './../util.js'
export class RequestFactory extends util.BaseAsyncObj{
  constructor(url){
    super()
    this.counter = 0;
    this.url = url
    try{
      XMLHttpRequest;
      this.send = (path, method, cb, data = undefined)=>{
        const xhr = new XMLHttpRequest();
        xhr.timeout = 0;
        xhr.responseType = "arraybuffer"
        xhr.onreadystatechange = ()=>{
          if(xhr.readyState === XMLHttpRequest.DONE){
            if(xhr.status === 200) {
              cb(new Uint8Array(xhr.response));
            } else console.error("Failed (non 200) response")
          }
        }
        xhr.open(method,this.url+path)
        if(data){
          xhr.setRequestHeader("Content-Type", "application/octet-stream");
        }
        xhr.send(data)
      }
      this.settle()
    }catch{
      let parts = url.match(/(^[^/\\]+)(.*)/)
      if(parts[2][parts[2].length-1]!="/"){
        //parts[2]+="/"
      }
      import('https').then(https=>{
        this.send = (path, method, cb, data = undefined)=> {
          const options = {
            hostname: parts[1],
            port: 443,
            path: parts[2]+path,
            method: method
          };
          console.log(options.hostname+options.path)
          const req = https.request(options, (res) => {
            let data = new util.StreamCache();
            res.on('data', (chunk) => {
              data.enqueue(new Uint8Array(chunk));
            });
            res.on('end', () => {
              cb(data.readnow(0))
            });
          });
          req.on('error', (e) => {
            console.error(`Request error: ${e.message}`);
          });
          if(data){
            req.write(data)
          }
          req.end();
        }
        this.settle()
      })
    }
  }
  send(path, method, cb, data = undefined){
    this.when(()=>this.send(path, method, cb, data))
  }
}