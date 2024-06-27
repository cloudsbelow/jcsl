import * as su from './src/util/SerialUtils.js'
import {PTXFile, BaseCudaContext} from './src/base/BaseCudaContext.js'
import { GraphContext } from './src/graph/GraphContext.js';

jcsl = {su, PTXFile, BaseCudaContext, GraphContext}

console.log("Loaded modules")
const script = document.createElement('script');
script.src = './browser/main.js';
document.body.appendChild(script);
