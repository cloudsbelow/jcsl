# jcsl
Javascript/CUDA serialization link (Not ~completely~ a joke) WIP/DYSFUNCTIONAL RIGHT NOW
A library to trivialize running the funnest programming language from the friendliest one
Run CUDA kernels from the comfort of JS. No need to sweat the async stuff, it's all handled.


to use: (I would not reccomend using this yet. if you have somehow run across it, keep running)
really, if you're using this, its prolly easier to just figure it out by codegazing

NVCC child.cu into build/child.exe
you probably want NVCC on your path for whenever I get around to JIT of .cu instead of just .ptx

For browser:
Include jcsl.js as module, write your code in browser/main (will be made more proper eventually)
run server.js

For server:
Include jcsl.js as module and add some fake objects to neutralize the whole automatic startup thing
(I did it for ctrl-i debugging because the browser consoles are so powerful and strong and wonderful
to work with, will probably remove once this gets somewhat finished)
make a base context with a su.Cprocess instead of Rprocess, everything else is exactly the same
(Remote api's 1-1 match with local API's)


