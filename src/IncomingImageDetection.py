from PIL import Image
import numpy as np
import cv2
from random import sample
from pyspark import SparkContext
import os
from pyspark.ml.feature import BucketedRandomProjectionLSHModel
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from sqlalchemy import create_engine
import pandas as pd
import psycopg2
import boto.s3.connection
import boto
import io
import sys
os.environ['PYTHONPATH']='python3'

user_path = sys.argv[1]

def detection(path):
    sc = SparkContext()
    spark = SparkSession(sc)

    #for incoming image from flask and saved in ec2, preprocessing it separately
    incoming_img = Image.open(path)
    incoming_img = np.asarray(incoming_img.resize((64,64)))
    incoming_img = incoming_img.reshape(64*64*3,order = 'C')
    incoming_img = incoming_img.astype(np.int16)
    incoming_img = Vectors.dense(incoming_img)
    incoming_img = [('test1.jpg',incoming_img)]

    # Load LSH model and calculate the hash value for incoming image
    incoming_df = spark.createDataFrame(incoming_img, ["id", "features"])
    BRP_model = BucketedRandomProjectionLSHModel.load("s3a://lsh-model/BRP_model.model")
    incoming_lsh = BRP_model.transform(incoming_df)
    incoming_lsh = incoming_lsh.drop('features')
    incoming_lsh = incoming_lsh.toPandas()

    incoming_lsh_list = []
    for i in incoming_lsh.ix[0,'hashes']:
        j = list(i)[0]
        incoming_lsh_list.append(j)

    # load existing hash values into EC2 and convert to the list format
    engine = create_engine('postgresql://xiaoxuan:**********@3.93.121.43:5432/xiaoxuan') #'**********'is the password of database
    existing_hash = engine.execute("SELECT * FROM hashes")
    existing_hash = list(existing_hash)

    existing_hash_preprocessed = []
    for i in range(0,len(existing_hash)):
        temp_id = existing_hash[i][0]
        temp_hash = existing_hash[i][1].split(';')
        for j in range(0,len(temp_hash)):
            temp_hash[j] = float((temp_hash[j]))
        temp = [temp_id,temp_hash]
        existing_hash_preprocessed.append(temp)

    # Calculate euclidean distance between hash value of incoming image and existing images
    def euclidean_distance(list1,list2):
        sum = 0
        for i in range(0,len(list1)):
            sum += (list1[i]-list2[i])**2
        return sum

    distance_list = []
    for i in existing_hash_preprocessed:
        hash_distance = euclidean_distance(incoming_lsh_list,i[1])
        id_and_distance = [i[0],hash_distance]
        distance_list.append(id_and_distance)
    distance_list = sorted(distance_list,key=lambda x:x[1])

    # Identify 6 nearest neighbors
    similar_image_id_list = []
    for i in range(0,6):
        similar_image_id_list.append(distance_list[i][0])

    # Write results(images) to specified user directory
    aws_access_key = '****************'  #your access key
    aws_secret_access_key = '******************'  #your secret access key
    conn = boto.connect_s3(aws_access_key, aws_secret_access_key)
    bucket = conn.get_bucket("celebrity-test")

    for i in range(0,len(similar_image_id_list)):
        key = bucket.get_key(similar_image_id_list[i])
        s = key.get_contents_as_string()
        pil_image = Image.open(io.BytesIO(s))
        opencvImage = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        cv2.imwrite(user_path+'result'+str(i+1)+'.jpg', opencvImage)


detection(user_path+'test1.jpg')
