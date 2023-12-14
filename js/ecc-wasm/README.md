# bchlib-wasm

This folder contains a Web Assembly (wasm) wrapper for the standalone python bchlib BCH error correction routines, which are used 
to perform error correction on the raw payload extracted from the watermark in order to add further resilience to the watermarked 
identifier embedded in the image.


## Build instructions

Emscripten is required to compile the wasm using the script.  It has been tested on macOs Emscripten 3.1.46.

First obtain bch.c and bch.h from the Python bchlib repo (https://github.com/jkent/python-bchlib/src)

Then run the compilations script

```
sh ./compile-wash.sh
```

The build results in artifacts:

```
bchlib-wasm.wasm
bchlib-wasm.js
```

both of which are used by the Javascript watermark encoder/decoder code in this repo.

## Note on deployment

Note that some browsers e.g. Chrome will require these artifacts to be hosted rather than on local disk in order to load them. 
One way to work around this is to locally host the demo JS code and this subfolder using 

```
cd ..
python -m http.server 8080
```





