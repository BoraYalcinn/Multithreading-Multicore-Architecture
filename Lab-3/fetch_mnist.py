# #!/usr/bin/env python3
#PC40 Lab 03 J.Gaber (gaber@utbm.fr)

import argparse, gzip, hashlib, os, shutil, urllib.request
FILES = {
 'train-images-idx3-ubyte.gz':'f68b3c2dcbeaaa9fbdd348bbdeb94873',
 'train-labels-idx1-ubyte.gz':'d53e105ee54ea40749a09fcbcd1e9432',
 't10k-images-idx3-ubyte.gz':'9fb629c4189551a2d022fa330f9573f3',
 't10k-labels-idx1-ubyte.gz':'ec29112dd5afa0611ce80d1b7f02629c',
}
MIRRORS=['https://ossci-datasets.s3.amazonaws.com/mnist/','https://storage.googleapis.com/cvdf-datasets/mnist/']
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--output',default='data/mnist'); a=ap.parse_args(); os.makedirs(a.output,exist_ok=True)
 for fn,md5 in FILES.items():
  gz=os.path.join(a.output,fn); raw=gz[:-3]
  if not os.path.exists(gz):
   last=None
   for base in MIRRORS:
    try: print('downloading',base+fn); urllib.request.urlretrieve(base+fn,gz); break
    except Exception as e: last=e
   else: raise RuntimeError(f'cannot download {fn}: {last}')
  h=hashlib.md5(open(gz,'rb').read()).hexdigest()
  if h!=md5: raise RuntimeError(f'checksum mismatch for {fn}: {h}')
  if not os.path.exists(raw):
   with gzip.open(gz,'rb') as src, open(raw,'wb') as dst: shutil.copyfileobj(src,dst)
  print('ready',raw)
if __name__=='__main__': main()
