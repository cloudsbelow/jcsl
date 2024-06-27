import fs from 'fs'
import http from 'http'
import * as su from './src/util/SerialUtils.js'

const contentTypes = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.png': 'image/x-icon'
}
const durls = {
  '/':"browser/index.html",
  '/favicon.ico':'browser/favicon.png'
}
function rejectReq(num, res, err){
  console.log("rejected a request with reason "+num+" for "+err);
  res.writeHead(num, { 'Content-Type': 'text/plain' });
  if(num == 404) res.end('not found');
  if(num == 500) res.end('server error');
}

let processes = {};
const f = new su.FileContextServer('f', "./")

const server = http.createServer((req, res) => {
  console.log("SERVING: "+ req.url, req.method);
  const spath = req.url.split('/');
  if(spath[1]=='r'){
    if(processes[spath[2]] == undefined && spath[3]=='open'){
      console.log("Opening new process with id: "+spath[2])
      processes[spath[2]] = new su.RProcessServer(spath[2],"./build/child.exe",()=>{
        console.log(Object.keys(processes).length+ " processes currently open")
        res.writeHead(200, "success")
        return res.end()
      },(process)=>{
        delete processes[process.id]
      })
      return;
    }
    return processes[spath[2]]?.processRequest(req,res)
  }
  if(spath[1]=='f'){
    return f.processRequest(req,res)
  }
  if(req.method === 'GET'){
    const file = durls[req.url]?durls[req.url]:req.url.slice(1);
    fs.access(file, fs.constants.R_OK, (err)=>{
      if(err) return rejectReq(404, res, err);
      const fstream = fs.createReadStream(file);
      res.writeHead(200, {'Content-Type': contentTypes[file.match(/\.[a-zA-Z]+$/)?.[0]]})
      fstream.pipe(res);
      fstream.on('close', ()=>{res.end()});
    });
  }
});
const port = 3000;
const address = '0.0.0.0'
server.listen(port, address, () => {
  console.log(`Server listening at http://${address}:${port}`);
});





