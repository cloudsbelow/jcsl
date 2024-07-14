//import * as su from './src/util/SerialUtils.js'
import {PTXFile, BaseCudaContext} from './src/base/BaseCudaContext.js'
import { GraphContext } from './src/graph/GraphContext.js';
import { FProcess } from './src/debug/fakes.js'
import { RProcessClient } from './src/util/serial/process.js';
import { CProcess } from './src/util/serial/process.js';

jcsl = {su, PTXFile, BaseCudaContext, GraphContext, FProcess, RProcessClient, CProcess}

console.log("Loaded modules")
const script = document.createElement('script');
script.src = './browser/main.js';
document.body.appendChild(script);
