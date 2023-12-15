# Copyright 2023 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.


from trustmark import TrustMark
from PIL import Image
import random,string,json
import c2pa_python as c2pa

tm=TrustMark(verbose=True, model_type='C')

watermarkid=''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

# encoding example
cover = Image.open('ufo_240.jpg').convert('RGB')
tm.encode(cover, watermarkid).save('out/ufo_240_C.png', exif=cover.info.get('exif'), icc_profile=cover.info.get('icc_profile'), dpi=cover.info.get('dpi'))


test_cert = open("keys/es256_certs.pem","rb").read()
test_key = open("keys/es256_private.key","rb").read()
sign_info = c2pa.SignerInfo("es256", test_cert, test_key, "http://timestamp.digicert.com")

manifest_json = json.dumps({
       "claim_generator": "python_trustmark/1.0",
       "title": "UFO Test Image",
       "thumbnail": {
           "format": "image/png",
           "identifier": "ufo_240_C.jpg"
       },
       "assertions": [
       {
           "label": "c2pa.soft-binding",
           "data": {
                 "alg": "trustmark-C",
                 "blocks": [
                       {
                          "scope": {
                          },
                          "value": "$watermarkid"
                       }
                 ]
            }
        },
        {
            "label": "c2pa.actions",
            "data": {
                "actions": [
                    {
                        "action": "c2pa.opened"
                    },     
                    {
                        "action": "c2pa.watermarked"
                    }     
                ]
            }
        }
        ]
})

c2pa_data = c2pa.sign_file("out/ufo_240_C_signed.png", output_dir+"/out.jpg", manifest_json, sign_info, 'out')


# decoding example
cover = Image.open('out/ufo_240_C_signed.png').convert('RGB')
wm_secret, wm_present = tm.decode(cover)
if wm_present:
  print(f'Extracted secret: {wm_secret}')
else:
  print('No watermark detected')

