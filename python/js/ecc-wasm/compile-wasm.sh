emcc bchlib-wasm.c bch.c -o bchlib-wasm.html -sEXPORTED_FUNCTIONS=["_init","_inplacedecode","_inplaceencode","UTF8ToString","stringToUTF8Array","lengthBytesUTF8","_malloc","_free"] -sEXPORTED_RUNTIME_METHODS=["ccall","cwrap","setValue"] 

