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
      resultArray.push((c & 0x80)>0);
      resultArray.push((c & 0x40)>0);
      resultArray.push((c & 0x20)>0);
      resultArray.push((c & 0x10)>0);
      resultArray.push((c & 0x8)>0);
      resultArray.push((c & 0x4)>0);
      resultArray.push((c & 0x2)>0);
      resultArray.push((c & 0x1)>0);
  }
  return resultArray;
}

function booleansToASCII7(booleanArray) {
  if (booleanArray.length % 7 !== 0) {
    throw new Error('The input array length must be a multiple of 7.');
  }

  const resultArray = [];
  for (let i = 0; i < booleanArray.length; i += 7) {
    const byte = booleanArray.slice(i, i + 7).reduce(function (acc, bit, index) {
      if (bit) {
        // Set the corresponding bit in the byte
        acc |= (1 << (6 - index));
      }
      return acc;
    }, 0);
    
    resultArray.push(byte);
  }

  return String.fromCharCode.apply(null, resultArray);
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

const toHexString = (bytes) => {
  return Array.from(bytes, (byte) => {
    return ('0' + (byte & 0xff).toString(16)).slice(-2);
  }).join('');
};

async function run() {

  try {
    // Load model
    log('******decoding********');
    startTime = new Date();
    log('Loading decode model...');
    const session = await ort.InferenceSession.create('models/decoder_C.onnx');
    timeElapsed = new Date() - startTime;
    log('Decode model loaded in ' + timeElapsed/1000 + ' seconds');


    // prepare dummy input data
    const dims = [1, 3, 256, 256];
    const size = dims[0] * dims[1] * dims[2] * dims[3];
    //    const inputData = Float32Array.from({ length: size }, () => Math.random());
     const inputData = new Float32Array(size);

    startTime = new Date();
    log('Loading watermarked test image...');
    const imageUrl = 'ufo_240_C.png';


    ready=false;
    loadAndResizeImage(imageUrl, inputData, (error, resizedImageUrl) => {
    if (error) {
       log(error.message);
    } else {
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
    const feeds = { image: new ort.Tensor('float32', inputData, dims) };
    timeElapsed = new Date() - startTime;

    // feed inputs and run
    const results = await session.run(feeds);
    watermarkfloat=results['202'].data ;
    watermarkbool=watermarkfloat.map(function (floatValue) {
      // Check the sign of the float and convert to boolean
      return floatValue >= 0;
    });
    log('Watermark extracted in ' + timeElapsed + ' millis');

    watermark=ECCWatermark(watermarkbool);

    if (watermark) {
       log('Watermark present -> [' +watermark+']');
    }
    else {
       log('No watermark present');
    }

  } catch (e) {
    console.log(e);
  }

  log('******Decoding Done********');

}

function ECCWatermark(watermarkbool) {

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
       watermark = booleansToASCII7(watermarkbool_corrected);
   }
   else {
       watermark = '';
   }

   return (watermark);

}
run();
