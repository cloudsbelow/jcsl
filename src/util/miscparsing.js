//arguments with one dash (-r) take one argument
//arguments with two dashes (--r) take arguments until the next flag
//arguments with numberized dash (-n-r) take n arguments
export function parsecmd(args){
  let ret = {
    unflagged:[]
  }
  let gathercount = 0;
  let gather = undefined
  args.forEach(arg => {
    if(gathercount){
      if(gathercount==-1){
        if(arg[0]=='-'){
          gathercount=0
        } else {
          return gather.push(arg)
        }
      } else {
        if(arg[0] == '-') throw Error("parse error - not enough after -n-");
        else {
          gathercount--;
          if(typeof gather == 'object') return gather.push(arg);
          else return ret[gather] = arg;
        }
      }
    } 
    const match = arg.match(/^-(\d*)(-?)(\w+)$/)
    if(match){
      gather = match[3]
      gathercount = 1;
      if(match[2]){
        gather = []
        gathercount = -1
        ret[match[3]] = gather
      }
      if(match[1]) gathercount = parseInt(match[1]);

    } else {
      ret.unflagged.push(arg)
    }
  });
  return ret
}