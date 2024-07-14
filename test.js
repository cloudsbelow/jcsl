import { RequestFactory } from "./src/util/serial/genericreq.js";
import * as util from "./src/util/util.js"

console.log("hi")
const c = new RequestFactory("example.com")
c.send("","GET",(data)=>{
  console.log(new TextDecoder().decode(data))
})
