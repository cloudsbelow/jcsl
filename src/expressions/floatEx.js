//autograd stuff for floating point operations

const TOK = '([@#$]\\w+|#?&[@#]?\\w+&[#@]?\\w+)'; //([@#%]\w+|&#?\w+&#?\w+)
const TOKRE = new RegExp(TOK,'g');
const ADDRE = new RegExp(`(^|(?<=[(,]))${TOK}?([+-]${TOK})+($|(?=[),]))`, 'g')
const MULTRE = new RegExp(`(^|(?<=[(+,-]))(${TOK}[\\/\\*])+${TOK}($|(?=[)+,-]))`, 'g')
const PARENRE = new RegExp(`[a-z]*\\(${TOK}(,${TOK})*\\)`, 'g')
const CONSTRE = /(^|(?<=[(\.+\-*\/\^,]))\d+(\.\d+)?($|(?=[)\.+\-*\/\^,]))/
const MAXDEPTH = 200;

const HPTOK = '!T!'
  
function commasBefore(ex, item){
  let pos = []
  let idx =  -1
  while((idx = ex.indexOf(item))!=-1){
    pos.push(ex.slice(0,idx).match(/,/g)?.length??0)
    ex=ex.replace(item,"")
  }
  return pos;
}
function keyIntersect(...o){
  const idx = Object.keys(o[0]).length<Object.keys(o[1]).length?1:0;
  const res = {}
  Object.keys(o[1-idx]).forEach((key)=>{
    if(o[idx][key] !== undefined) res[key] = true
  })
  return res
}

const ops = {
  2:{
    f: (ex)=>ex,
    d: (ex, item)=>{
      const oc = ex.match(new RegExp(`[+-]?${item}`,'g'))
      return oc.length-2*oc.filter(s=>s[0]=='-').length+"*"+HPTOK
    }
  },
  1:{
    f: (ex)=>ex,
    d: (ex, item)=>{
      const oc = ex.match(new RegExp(`[\\/\\*]?${item}`,'g'))
      const p = oc.length-2*oc.filter(s=>s[0]=='/').length
      if(p>0) return `${p}*${HPTOK}*${ex.replace(item, 1)}`
      if(p<0) return `${p}*${HPTOK}*${ex}/${item}`;
      return 0
    }
  },
  exp:{ //exp a -> returns e^a (like what else)
    args:1,
    d:(args, idx)=>`exp(${args[0]})*${HPTOK}`
  },
  select:{ //select p,a,b -> if p>=0, returns a, otherwise b (ternary type)
    args:1,
    d:(args, idx)=>{
      if(idx == 0) return 0;
      if(idx == 1) return `select(${args[0]}, ${HPTOK}, 0)`
      if(idx == 2) return `select(${args[0]}, 0, ${HPTOK})`
    }
  },
  clamp:{ //clamp x,a,b -> returns x clamped between a and b
    args:3,
    d:(args, idx)=>{
      if(idx == 0) return `select(${args[0]}-${args[1]},select(${args[2]-args[0]},${HPTOK},0),0)`;
      if(idx == 1) return `select(${args[0]}-${args[1]},0,${HPTOK})`
      if(idx == 2) return `select(${args[2]}-${args[0]},0,${HPTOK})`
      throw Error("?")
    }
  },
  max:{ //max a,b -> returns max of a and b
    args:2,
    d:(args, idx)=>{
      if(idx == 0) return `select(${args[0]}-${args[1]},${HPTOK},0)`;
      if(idx == 1) return `select(${args[0]}-${args[1]},0,${HPTOK})`;
    }
  },
  min:{ //min a,b -> returns min of a and b
    args:2,
    d:(args, idx)=>{
      if(idx == 0) return `select(${args[0]}-${args[1]},0,${HPTOK})`;;
      if(idx == 1) return `select(${args[0]}-${args[1]},${HPTOK},0)`;
    }
  },
  ln:{
    args:1,
    d:(args, idx)=>`${HPTOK}/${args[0]}`
  },
  0:{
    f:(ex)=>ex,
    d:(ex, item)=>{
      const pos = commasBefore(ex, item)
      const args = ex.match(TOKRE)
      const op = ex.match(/[a-z]*/)[0]
      return pos.map((idx)=>ops[op].d(args,idx)).reduce((a,b)=>{a+"+"+b})
    }
  },
  "":{
    f:(ex)=>ex,
    d:(ex, item)=>HPTOK
  }
}

class FloatEx{
  constructor(text){
    this.symmap={}
    this.microt = {} 
    this.macrot = {}
    this.manual = {}
    this.symcounter = 0

    const lines = text.replaceAll(/\/\/.*$/gm,"").match(/^.+$/gm).map(s=>s.replaceAll(/\s/g,""))
    console.log(lines)
    lines.map(str=>str.split('=')).forEach(([sym,f])=>{
      this.addline(sym, f)
    })
  }
}

FloatEx.prototype.symadd = function(expr, type){
  if(this.symmap[expr]) return this.symmap[expr]
  const sym = '#'+(++this.symcounter);
  const s = {
    expr:expr, t: type,
    in:[...new Set(type==-1?[]:expr.match(TOKRE))]
  }
  this.symmap[expr] = sym
  this.microt[sym] = s
  return sym
}
FloatEx.prototype.derivtag = function(to,from){return `#&${to}&${from}`}
FloatEx.prototype.derivadd = function(expr, to, from){
  const sym = this.derivtag(to, from)
  if(this.microt[sym]) return this.microt[sym].s;
  const fl = this.symReduce(expr)
  const s = {
    expr:fl, s:fl, t:-2, in:[fl], oin:[...new Set(expr.match(TOKRE)??[])], oex:expr,
  }
  this.microt[sym] = s
  return fl
}

FloatEx.prototype.recursiveIn = function(list, set={}){
  list?.forEach(li=>{
    if(li.match(/[a-zA-Z_]/,'g') && !set[li]){
      if(!this.macrot[li]){
        if(li.match(/^#?&[@#]?\w+&[#@]?\w+$/g)){
          let parts = li.match(/(?<=&)\w+/g)
          if(!set.flag)set.flag = [];
          set.flag.push([li, this.genpartial("@"+parts[0],"@"+parts[1])])
        } else {
          throw Error('Undefined symbol '+li);
        }
      }
      this.recursiveIn(this.macrot[li].in, set)
    }
    set[li]=true
  })
  return set;
}
FloatEx.prototype.recursiveMicroIn = function(node, set=[], done={}){
  if(done[node]) return;
  done[node]=true;
  if(node[0]=="@"||node[0]=="&"){
    set.push(node)
    return set
  }
  this.microt[node].in.forEach(li=>{
    this.recursiveMicroIn(li, set, done)
  })
  return set;
}

FloatEx.prototype.gendirpartial = function(to, from){
  if(this.macrot[to].in.indexOf(from) == -1) throw Error('Cannot generate direct partial of non-direct relation');
  if(this.microt[`#&${to}&${from}`]) return this.microt[`#&${to}&${from}`].s;
  let affected = {}
  const gather = (n, done={})=>{
    // to take care of higher order derivatives because I am mentally unwell!
    while(this.microt[n]?.t==-2){
      n = this.microt[n].alt?this.microt[n].alt:(this.microt[n].alt = this.symReduce(this.microt[n].expr))
    }
    if(done[n]) return affected[n];
    done[n] = true;
    if(n[0] == '@'){
      if(n == from) return affected[n] = {end:true, out:[]}
    } else {
      const c = this.microt[n].in.filter(s=>gather(s,done)?.out.push(n))
      if(c.length>0) return affected[n]={in:c,out:[]} 
    }
  }
  gather(this.macrot[to].s)
  const propegate = (n,done={})=>{
    if(n == this.macrot[to].s) return 1;
    if(done[n]) return this.derivadd(null, to, n);
    done[n] = true
    let ex = ""
    affected[n].out.forEach((s)=>{
      const sym = propegate(s, done)
      //console.log(s,sym)
      let expr = ops[this.microt[s].t].d(this.microt[s].expr,n).replaceAll(HPTOK, sym)
      ex+=(ex==""?"":"+")+expr.replaceAll(/((^|(?<=[(+,\-\*\/]))1\*)|(\*1($|(?=[)+,\-\*\/])))/g,"")
    })
    return this.derivadd(ex, to, n)
  }
  return propegate(from)
}
//All of this is so much more of a pain than it needs to be due to
//needing to generate partial derivatives from partially specified
//manual derivatives. Unironically increases complexity by like 8x
FloatEx.prototype.genpartial = function(to, from){
  if(this.macrot[(to+from).replaceAll('@',"&")]){
    return this.macrot[(to+from).replaceAll('@',"&")].s;
  }
  if(!this.macrot[to].inall[from]){
    console.log('nderiv')
    const ret = (this.macrot[(to+from).replaceAll('@',"&")]={
      s:this.symadd('0',-1), in:[], inall:{}
    })
    return ret.s 
  }
  let gnodes = {}
  let edges = {}
  const explore1 = (node)=>{
    //if(node.match(/@\d+/)) return;
    if(gnodes[node]) return
    const fnodes = this.macrot[node]?.in.filter((s)=>{
      return s==from || this.macrot[s]?.inall[from]
    })
    fnodes?.forEach(s=>{
      edges[node+s]=1
      explore1(s)
      gnodes[s].out.push(node)
    })
    gnodes[node] = {
      in:fnodes??[], out:[], manualin:[]
    }
  }
  explore1(to)
  const aggbyat = (node, followat, set)=>{
    gnodes[node][followat].forEach(li=>{
      if(!set[li]) aggbyat(li, followat, set)
      set[li]=true
    })
  }
  const markfilter = (node, filter, done={})=>{
    if(!done[node]){
      done[node]=true
      gnodes[node].out.forEach((s)=>{
        if(!filter[s]) return
        markfilter(s, filter, done)
        if(edges[s+node] & 2) throw Error("Manual derivatives overlap")
        edges[s+node] |= 2;
      })
    }
  }
  const iscontained = (node, att, filter, end, done={})=>{
    if(node == end || done[node]) return true;
    if(!filter[node]) return false;
    done[node]=true;
    return gnodes[node][att].map(s=>iscontained(s, att, filter, end, done)).reduce((a,b)=>a&&b)
  }
  const nobypass = (node, filter, end, done={})=>{
    if(node == end) return false;
    if(!filter[node] || done[node]) return true;
    done[node] = true;
    return gnodes[node].in.map(s=>{
      if(edges[node+s] & 4) return true;
      return nobypass(s, filter, end, done)
    }).reduce((a,b)=>a&&b)
  }
  Object.keys(gnodes).forEach((node)=>{
    if(this.manual[node]){
      aggbyat(node, 'out', gnodes[node].outall={})
      this.manual[node].forEach((s)=>{
        if(gnodes[node].outall[s]){
          aggbyat(s,'in',gnodes[s].inall={})
          let mark = keyIntersect(gnodes[node].outall, gnodes[s].inall)
          mark[s]=true;
          markfilter(node, mark)
          gnodes[s].in.forEach(sy=>{
            if(iscontained(sy, 'in', mark, node)) edges[s+sy] |= 4;
          })
          gnodes[node].out.forEach(sy=>{
            if(iscontained(sy, 'out', mark, s)) edges[sy+node] |=4;
          })
          if(!nobypass(s, mark, node)) throw Error("badness");
          else gnodes[s].manualin.push(node)
        }
      })
    }
  })
  
  //this is all we would need (more even) if no manual derivatives :/
  let fnodes = {}
  const gather = (n)=>{
    if(fnodes[n]!==undefined) return fnodes[n];
    if(n == from) return fnodes[n] = [];
    const c1 = gnodes[n].in.filter(s=>!(edges[n+s]&4) && gather(s))
    gnodes[n].manualin.forEach(s=>gather(s))
    fnodes[n] = (c1.length+gnodes[n].manualin.length>0)?[]:false
    if(fnodes[n]){
      c1.forEach(s=>fnodes[s].push([1,this.gendirpartial(n,s),n]))
      gnodes[n].manualin.forEach(s=>fnodes[s].push([2,this.macrot[(n+s).replaceAll('@',"&")].s,n]))
    }
    return fnodes[n]
  }
  gather(to)
  const finish = (n, done={})=>{
    if(done[n]) return done[n]
    if(n == to) return done[n] = '1'
    const c = fnodes[n].map(([type, partial, out])=>{
      return partial+"*"+finish(out, done)
    })
    //console.log(n,c,fnodes[n])
    return done[n] = this.derivadd(c.reduce((a,b)=>a+"+"+b), to, n)
  }
  const fin = finish(from);
  const inputs = this.recursiveMicroIn(fin);
  const inall = this.recursiveIn(inputs)
  if(inall.flag) throw Error('Something happened lmao good luck')
  this.macrot[(to+from).replaceAll('@','&')] = {
    s:fin, in:inputs, inall:inall
  }
  return fin
}

FloatEx.prototype.compileout = function(targets){
  let ctr = 0;
  let rename = {};
  const tocompilename = (str)=>{
    if(rename[str]) return rename[str]
    if(str[0]=='#'){
      if(this.microt[str].t !== -1) 
        return rename[str]='v'+(++ctr);
      else return this.microt[str].expr;
    }
    if(str.match(/^@\d+$/)) return rename[str]='loaded_'+str.match(/\d+/g)[0]
    return rename[str] = tocompilename(this.macrot[str].s)
  }
  const tobasename = (str)=>{
    if(str[0] == '#' || str.match(/^@\d+$/)) return str;
    return tobasename(this.macrot[str].s)
  }
  let enqueued = {}
  let queue = []
  const enqdep = (n)=>{
    n = tobasename(n)
    console.log(n)
    if(n[0] != '@' && this.microt[n].t == -1) return;
    if(enqueued[n]) return;
    enqueued[n] = true
    this.microt[n]?.in.forEach(enqdep)
    queue.push(n)
  }
  targets.forEach(enqdep)
  console.log(queue)

  let prog = `//Autogenerated code\n`
  for(let i=0; i<queue.length; i++){
    let s = queue[i]
    if(s[0]=='@') continue;
    let ex = this.microt[s].ex??this.microt[s].expr
    let m = null
    while(m=ex.match(TOKRE)){
      ex=ex.replace(m[0],tocompilename(m[0]))
    }
    prog+=`float ${tocompilename(s)} = ${ex};\n`
  }


  return prog;
}

FloatEx.prototype.symReduce = function(f){
  if(this.symmap[f]) return this.symmap[f]
  let forig=f;
  let constval; //scrub constants
  for(let i=0; i<MAXDEPTH && (constval = f.match(CONSTRE)?.[0]); i++){
    f=f.replace(new RegExp('(^|(?<=[(\\.+\\-*\\/\\^,]))'+constval),this.symadd(constval, -1))
  }
  for(let i=0; i<MAXDEPTH; i++){
    let t=0;
    let m = f.match(PARENRE)
    if(m===null && ++t) m = f.match(MULTRE)
    if(m===null && ++t) m = f.match(ADDRE)
    if(m===null){
      let final = f.match(new RegExp(`^${TOK}$`))[0]
      if(final){
        return this.symmap[forig]=final
      } else {
        throw console.error("Parse error", f)
      }
    }
    m.forEach((s)=>{
      f=f.replace(s,this.symadd(s,t))
    })
  }
}
FloatEx.prototype.addline = function(sym, f){
  let macroin = [...new Set(f.match(TOKRE))]
  let inall = this.recursiveIn(macroin)
  if(inall.flag){
    inall.flag.forEach(([ex, link])=>f=f.replaceAll(ex,link))
  }
  let s = this.symReduce(f)
  if(inall.flag){
    macroin = this.recursiveMicroIn(s)
    inall = this.recursiveIn(macroin)
  }
  this.macrot[sym]={
    s:s, in:macroin, inall:inall
  };
  
  if(sym.match(/&\w+&\w+/g)){
    this.manual[sym]=true
    const from = "@"+sym.match(/\w+$/)[0]
    if(!this.manual[from]) this.manual[from] = []
    this.manual[from].push("@"+sym.match(/\w+/)[0])
  }
}







let scr = `
// do thing
@1 = @2*@2
@W = @1
`
//let f = new FloatEx(scr)
//console.log(f.compileout(['@W']))
//2@1+1+@2*exp(@1*@2)