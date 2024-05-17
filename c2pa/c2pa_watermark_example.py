import random,os
from PIL import Image
import json
import struct

from trustmark import TrustMark

from PIL import Image

TM_SCHEMA_CODE=TrustMark.Encoding.BCH_4

def uuidgen(bitlen):

    id = ''.join(random.choice('01') for _ in range(bitlen))
    return id


def embed_watermark(img_in, img_out, watermarkID, tm):
   cover = Image.open(img_in)  
   rgb=cover.convert('RGB')  
   has_alpha=cover.mode== 'RGBA'
   if (has_alpha):
      alpha=cover.split()[-1]
   encoded=tm.encode(rgb, watermarkID, MODE='binary')
   params={
      "exif":cover.info.get('exif'),
      "icc_profile":cover.info.get('icc_profile'),
      "dpi":cover.info.get('dpi')
   }
   not_none_params = {k:v for k, v in params.items() if v is not None}
   encoded.save(img_out, **not_none_params)

def build_manifest(watermarkID, img_in):

       assertions=[]
       assertions.append(build_softbinding('com.adobe.trustmark.Q',str(TM_SCHEMA_CODE)+"*"+watermarkID))
       actions=[]
       act=dict()
       act['action']='c2pa.watermarked'
       actions.append(act)
       
       manifest=dict()
       manifest['claim_generator']="python_trustmark/c2pa"
       manifest['title']="Watermarked Image"
       manifest['thumbnail']=dict()
       manifest['ingredient_paths']=[img_in]

       ext=img_in.split('.')[-1]
       manifest['thumbnail']['format']="image/"+ext
       manifest['thumbnail']['identifier']=img_in
       manifest['assertions']=assertions
       manifest['actions']=actions
       
       return manifest

def build_softbinding(alg,val):       
       sba=dict()
       sba['label']='c2pa.soft-binding'
       sba['data']=dict()
       sba['data']['alg']=alg
       sba['data']['blocks']=list()
       blk=dict()
       blk['scope']=dict()
       blk['value']=val
       sba['data']['blocks'].append(blk)
       return sba

def manifest_add_signing(mf):
       mf['alg']='es256'
       mf['ta_url']='http://timestamp.digicert.com'
       mf['private_key']='keys/es256_private.key'
       mf['sign_cert']='keys/es256_certs.pem'
       return mf

def manifest_add_creator(mf,name):
       cwa=dict()
       cwa['label']='stds.schema-org.CreativeWork'
       cwa['data']=dict()
       cwa['data']['@context']='https://schema.org'
       cwa['data']['@type']='CreativeWork'
       author=dict()
       author['@type']='Person'
       author['name']=name
       cwa['data']['author']=[author]
       mf['assertions'].append(cwa)
       return mf
       

def main() :

   img_in='example.jpg'
   img_out='example_wm.jpg'
   img_out_signed='example_wm_signed.jpg'
	
   # Generate a random watermark ID
   tm=TrustMark(verbose=True, model_type='Q', encoding_type=TM_SCHEMA_CODE)
   bitlen=tm.schemaCapacity()
   id=uuidgen(bitlen)

   # Encode watermark
   embed_watermark(img_in, img_out, id, tm)

   # Build manifest
   mf=build_manifest(id, img_in)
   mf=manifest_add_creator(mf,"Walter Mark")
   mf=manifest_add_signing(mf)

   
   fp=open('manifest.json','wt')
   fp.write(json.dumps(mf, indent=4))
   fp.close()
   os.system('c2patool '+img_out+' -m manifest.json -f -o '+img_out_signed)
         


   # Check watermark present
   stego = Image.open(img_out_signed).convert('RGB')
   wm_id, wm_present, wm_schema = tm.decode(stego, 'binary')
   if wm_present:
      print('Watermark detected in signed image')
      if wm_id==id:
         print('Watermark is correct')
         print(id)
      else:
         print('Watermark does not match!')
         print(id)
         print(wm_id)
   else:
       print('No watermark detected!')   



if __name__ == "__main__":
    main()
 

