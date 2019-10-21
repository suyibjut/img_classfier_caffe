#!/usr/bin/env python
import sys
sys.path.append('/opt/caffe/python')
import numpy as np
import sys, caffe

if len(sys.argv) != 3:
    print "Usage: python bin2npy.py mean.binaryproto mean.npy"
    sys.exit()

blob = caffe.proto.caffe_pb2.BlobProto()
bin_mean = open(sys.argv[1], 'rb').read()
blob.ParseFromString(bin_mean)
arr = np.array(caffe.io.blobproto_to_array(blob))
npy_mean = arr[0]
print 'channel:', len(npy_mean), 'row:', len(npy_mean[0]), 'col:', len(npy_mean[0][0])
np.save(sys.argv[2], npy_mean)
