import io
import boto
from PIL import Image
import numpy as np
import cv2
from random import sample
from pyspark import SparkContext
import os
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col
from pyspark.sql.types import *
os.environ['PYTHONPATH']='python3'

#conf=SparkConf().setAppName("lsh_test").setMaster("spark://10.0.0.8:7077")

aws_access_key = 'AKIAJDOXOTJNGPA6WMQQ'
aws_secret_access_key = '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'
conn = boto.connect_s3(aws_access_key, aws_secret_access_key)
bucket = conn.get_bucket("picture-s3")


file_names = []
for key in bucket.list():
    file_names.append(str(key.name.encode('utf-8'))[2:-1])


def preprocessing_rdd(file_name):
        key = bucket.get_key(file_name)
        s = key.get_contents_as_string()
        img = Image.open(io.BytesIO(s))
        normal_size = (64, 64)
        img = np.asarray(img.resize(normal_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape(64*64,order = 'C')
        img = img.astype(np.int16)
        img = Vectors.dense(img)
        return img


file_names_rdd = sc.parallelize(file_names)

imgs_rdd = file_names_rdd.map(lambda file_name: (file_name, preprocessing_rdd(file_name)))

df = spark.createDataFrame(imgs_rdd, ["id", "features"])

brp = BucketedRandomProjectionLSH(inputCol="features",
                                  outputCol="hashes",
                                  bucketLength=30.0,
                                  numHashTables=50)

BRP_model = brp.fit(df)

#Feature Transformation
print("The hashed dataset where hashed values are stored in the column 'hashes':")
lsh = BRP_model.transform(df)
lsh.show()
