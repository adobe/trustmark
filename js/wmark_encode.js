/* 
 Copyright 2022 Adobe
 All Rights Reserved.

 NOTICE: Adobe permits you to use, modify, and distribute this file in
 accordance with the terms of the Adobe license agreement accompanying
 it. 
 */

function loadAndResizeImage(url, inputData, callback) {
  const img = new Image();


  img.onload = function () {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // Resize the image to 256x256 pixels
    canvas.width = 256;
    canvas.height = 256;
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height, {colorSpace: "srgb"});

    // Copy the image data into the 'inputData' array
    let j=0;
    let page=canvas.width*canvas.height;
    let twopage=canvas.width*canvas.height*2;
    for (let i = 0; i < canvas.width*canvas.height*4; i+=4) {
       // Normalize the pixel values to the range [0, 1]
       let b= (2.0*(imageData.data[i] / 255.0))-1.0;
       let g= (2.0*(imageData.data[i+1] / 255.0))-1.0;
       let r= (2.0*(imageData.data[i+2] / 255.0))-1.0;
       inputData[j] = r;
       inputData[j+page] = g;
       inputData[j+twopage] = b;
       j++;
    }


    // Convert the resized image to a data URL
    const resizedImageURL = canvas.toDataURL('image/png');

    // Execute the callback with the resized image URL
    callback(null, resizedImageURL);
  };

  img.onerror = function () {
    callback(new Error('Failed to load image'));
  };

  img.src = url;

}

function uint8ToBoolean(u8array) {

  const resultArray=[];

  for (let i=0; i<u8array.length; i++) {
      c=u8array[i];
      resultArray.push(((c & 0x80)>0)?1:0);
      resultArray.push(((c & 0x40)>0)?1:0);
      resultArray.push(((c & 0x20)>0)?1:0);
      resultArray.push(((c & 0x10)>0)?1:0);
      resultArray.push(((c & 0x8)>0)?1:0);
      resultArray.push(((c & 0x4)>0)?1:0);
      resultArray.push(((c & 0x2)>0)?1:0);
      resultArray.push(((c & 0x1)>0)?1:0);
  }
  return resultArray;
}

function booleansToASCII8(booleanArray) {
  if (booleanArray.length % 8 !== 0) {
    throw new Error('The input array length must be a multiple of 8.');
  }

  const resultArray=[];
  let pos=0;
  for (let i = 0; i < booleanArray.length; i += 8) {
    const byte = booleanArray.slice(i, i + 8).reduce(function (acc, bit, index) {
      if (bit) {
        // Set the corresponding bit in the byte
        acc |= (1 << (7 - index));
      }
      return acc;
    }, 0);
    
    resultArray.push(byte);
  }

  return String.fromCharCode.apply(null, resultArray);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function log(msg) {

   const divElement = document.createElement("div");
   divElement.textContent = new Date().toLocaleTimeString()+' '+msg.toString();
   document.body.appendChild(divElement);
}

async function run() {

  try {

    log('******Encoding********');
    // Load model
    startTime = new Date();
    log('Loading Encoder model...');
    const session = await ort.InferenceSession.create('models/encoder_C.onnx');
    
    timeElapsed = new Date() - startTime;
    log('Model loaded in ' + timeElapsed/1000 + ' seconds');


    // prepare dummy input data
    const dims = [1, 3, 256, 256];
    const size = dims[0] * dims[1] * dims[2] * dims[3];
    const secret_len = 100; // 7 bytes are allowed + 5 bytes of ecc
    //    const inputData = Float32Array.from({ length: size }, () => Math.random());
    const inputData = new Float32Array(size);
    const boolean_secret = Float32Array.from({ length: secret_len }, () => Math.random()<0.5);
    
    // get the ECC bytes to this boolean secret
    boolean_secret_ecc = ECCWatermark(boolean_secret);

    startTime = new Date();
    log('Loading test image...');
    const imageUrl = 'ufo_240.jpg';

    ready=false;
    loadAndResizeImage(imageUrl, inputData, (error, resizedImageUrl) => {
    if (error) {
       log(error.message);
    } else {
        log('Input Image Display')
       const imgElement = document.createElement('img');
       imgElement.src = resizedImageUrl;
       document.body.appendChild(imgElement);
       timeElapsed = new Date() - startTime;
       log('Image loaded from web in ' + timeElapsed + ' millis');
       ready=true;

    }});

    while(!ready) {await sleep(100); }

    // prepare feeds. use model input names as keys.
    log('Run watermark model..');
    startTime = new Date();
    const feeds = { 'onnx::Concat_0': new ort.Tensor('float32', inputData, dims), 'onnx::Gemm_1': new ort.Tensor('float32', boolean_secret_ecc, [1, secret_len]) };
    timeElapsed = new Date() - startTime;

    // feed inputs and run
    const results = await session.run(feeds);
    image=results['261'].data; // this image is [1, 3, 256, 256] and scaled between -1 to 1. In order to display convert the image to rgb uint8

    log('Encoded Image Display')
    // display the watermarked image on the 
    var canvas = document.createElement('canvas');// Create a canvas element
    var ctx = canvas.getContext('2d');
    // Set the dimensions of the canvas based on the image array
    canvas.width = 256;
    canvas.height = 256;
    let page=canvas.width*canvas.height;
    let twopage=canvas.width*canvas.height*2;
    // Loop through the image array and draw each pixel on the canvas
    for (var row = 0; row < canvas.height; row++) {
      for (var col = 0; col < canvas.width; col++) {
        var b = (image[row*canvas.height+col]+1.0)*255.0/2.0;
        var g = (image[page+(row*canvas.height+col)]+1)*255.0/2.0;
        var r = (image[twopage+(row*canvas.height+col)]+1)*255.0/2.0;

        // Set the pixel color on the canvas
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(col, row, 1, 1);
      }
    }
    // Append the canvas to the body or any other HTML element
    document.body.appendChild(canvas);

    
    imageFloat=image.map(function (floatValue) {
      // Check the sign of the float and convert to boolean
      return floatValue;
    });

    //Sanity Check
    // after this decode the image and see the boolean accuracy
    const session_dec = await ort.InferenceSession.create('models/decoder_C.onnx'); 
    const feeds_dec = { image: new ort.Tensor('float32', imageFloat, dims) };
    // feed inputs and run
    const results_dec = await session_dec.run(feeds_dec);
    watermarkfloat=results_dec['202'].data ;
    watermarkbool=watermarkfloat.map(function (floatValue) {
      // Check the sign of the float and convert to boolean
      return floatValue >= 0;
    });
    // ecc decode
    decoded_watermark = ECCWatermark_decode(watermarkbool);
    // compute the accuracy of the extracted watermark and the one embedded
    const accuracy = decoded_watermark.slice(0,56).every((value, index) => value === boolean_secret_ecc[index]);

    if(accuracy){
      log('Sanity check passed');
    }
    else{
      log('Sanity check failed');
    }
    

  } catch (e) {
    console.log(e);
  }

}

function ECCWatermark(watermarkbool_orig) {

   // Convert raw watermark bits to wasm string
   watermarkbool = watermarkbool_orig.slice(0,96);
   let byt = booleansToASCII8(watermarkbool);
   outstr="";
   for (let i=0; i<7; i++) {
     n=byt[i].charCodeAt(0); 
     if (n>=128) {
       outstr+=String.fromCharCode(1);
       outstr+=String.fromCharCode(n-128);
     }
     else {
       outstr+=String.fromCharCode(0);
       outstr+=String.fromCharCode(n);
     }
   }

   // Run ECC
   var ptr_to_decoded=bch_encode(outstr,7,5);

   // If null there is no watermark / the errors are too severe
   if (ptr_to_decoded) {
       var js_array = Module.HEAPU8.subarray(ptr_to_decoded,ptr_to_decoded+12);
       watermarkbool_corrected=uint8ToBoolean(js_array).slice(0,96);
       watermarkbool_corrected.push(0,0,0,0); // pad the array to 100 size
       // convert to float32
       watermark = new Float32Array(watermarkbool_corrected);
   }
   else {
       watermark = watermarkbool_orig;
   }

   return (watermark);
}

function ECCWatermark_decode(watermarkbool) {

   // Convert raw watermark bits to wasm string
   watermarkbool=watermarkbool.slice(0,96);
   let byt=booleansToASCII8(watermarkbool);
   outstr="";
   for (let i=0; i<12; i++) {
     n=byt[i].charCodeAt(0); 
     if (n>=128) {
       outstr+=String.fromCharCode(1);
       outstr+=String.fromCharCode(n-128);
     }
     else {
       outstr+=String.fromCharCode(0);
       outstr+=String.fromCharCode(n);
     }
   }

   // Run ECC
   var ptr_to_decoded=bch_decode(outstr,12);

   // If null there is no watermark / the errors are too severe
   if (ptr_to_decoded) {
       var js_array = Module.HEAPU8.subarray(ptr_to_decoded,ptr_to_decoded+12);
       watermarkbool_corrected=uint8ToBoolean(js_array).slice(0,56);
   }
   return (watermarkbool_corrected);

}

run();