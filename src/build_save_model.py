import io
import boto
from PIL import Image
import numpy as np
import cv2
from random import sample
from pyspark import SparkContext
import os
from pyspark.ml.feature import BucketedRandomProjectionLSH,BucketedRandomProjectionLSHModel
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col
from pyspark.sql.types import *
from pyspark.sql import SparkSession
import psycopg2
import pandas as pd
from sqlalchemy import create_engine
import pickle
os.environ['PYTHONPATH']='python3'


sc = SparkContext()
spark = SparkSession(sc)

aws_access_key = '****************'  #your access key
aws_secret_access_key = '******************'  #your secret access key
conn = boto.connect_s3(aws_access_key, aws_secret_access_key)
bucket = conn.get_bucket("celebrity-test")

# get file names in the s3 bucket
file_names = []
for key in bucket.list():
    file_names.append(str(key.name.encode('utf-8'))[2:-1])

# Convert each image to a dense vector
def preprocessing_rdd(file_name):
        key = bucket.get_key(file_name)
        try:
            s = key.get_contents_as_string()
            img = Image.open(io.BytesIO(s))
            normal_size = (64, 64)
            img = np.asarray(img.resize(normal_size))
            img = img.reshape(64*64*3,order = 'C')
            img = img.astype(np.int16)
            img = Vectors.dense(img)
        except:
            img = Vectors.dense([0]*64*64*3)
        return img

file_names_rdd = sc.parallelize(file_names)
imgs_rdd = file_names_rdd.map(lambda file_name: (file_name, preprocessing_rdd(file_name)))

# Create a spark dataframe containing photo id and preprocessed image
df = spark.createDataFrame(imgs_rdd, ["id", "features"])

# Create model
brp = BucketedRandomProjectionLSH(inputCol="features",
                                  outputCol="hashes",
                                  bucketLength=15.0,
                                  numHashTables=100)

BRP_model = brp.fit(df)

# Save the model to s3
BRP_model.write().overwrite().save("s3a://lsh-model/BRP_model.model")
