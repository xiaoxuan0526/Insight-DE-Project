import io
import boto
from PIL import Image
import numpy as np
import cv2
from pyspark import SparkContext
import os
from pyspark.ml.feature import BucketedRandomProjectionLSH, BucketedRandomProjectionLSHModel
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import *
from pyspark.sql import SparkSession
import psycopg2
import pandas as pd
from sqlalchemy import create_engine
import pyspark.sql.functions as F
import pyspark.sql.types as T
os.environ['PYTHONPATH']='python3'

sc = SparkContext()
spark = SparkSession(sc)

aws_access_key = '****************'  #your access key
aws_secret_access_key = '******************'  #your secret access key
conn = boto.connect_s3(aws_access_key, aws_secret_access_key)
bucket = conn.get_bucket("celebrity-test")

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


# Create a spark dataframe containing photo id and preprocessed images
df = spark.createDataFrame(imgs_rdd, ["id", "features"])

BRP_model = BucketedRandomProjectionLSHModel.load("s3a://lsh-model/BRP_model.model")

lsh = BRP_model.transform(df)
lsh = lsh.drop('features')


# Transform the hash value into string format for storage
def to_str(x):
    y = ''
    for i in x:
        y += str(i[0])
        y += ';'
    y = y[:-1]
    return y

my_udf = F.UserDefinedFunction(to_str, T.StringType())
lsh = lsh.withColumn('hash', my_udf('hashes'))
lsh = lsh.drop('hashes').withColumnRenamed('hash', 'hashes')

# Combine distributed Spark DataFrame to a Pandas DataFrame
def _map_to_pandas(rdds):
    return [pd.DataFrame(list(rdds))]

def topas(df, n_partitions=None):
    if n_partitions is not None: df = df.repartition(n_partitions)
    df_pand = df.rdd.mapPartitions(_map_to_pandas).collect()
    df_pand = pd.concat(df_pand)
    df_pand.columns = df.columns
    return df_pand

lsh_pandas = topas(lsh)

'''
# Old method to save hash value into PostgreSQL(Inefficient, Slow)
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import psycopg2

engine = create_engine('postgresql://xiaoxuan:BUwxx005026@3.93.121.43:5432/xiaoxuan')
try:
    lsh.to_sql('lsh',engine,index=False,if_exists='replace')
except Exception as e:
    print(e)

'''

# save the calculated hashes to PostgreSQL
def to_pg(df, table_name, con):
    # setup sql engine
    pd_sql_engine = pd.io.sql.pandasSQL_builder(con)
    # create table on engine
    table = pd.io.sql.SQLTable(table_name, pd_sql_engine, frame=df, if_exists='replace')
    table.create()
    # setup string io
    data = io.StringIO()
    df.to_csv(data, header=False, index=False, sep = '$')
    data.seek(0)
    raw = con.raw_connection()
    curs = raw.cursor()
    # write data to table
    curs.copy_from(data, table_name, sep='$', columns=('id', 'hashes'))
    # commit
    curs.connection.commit()
    return data

engine = create_engine('postgresql://xiaoxuan:************@3.93.121.43:5432/xiaoxuan') # '************'is the passward of database
to_pg(lsh_pandas, 'hashes', engine)
engine.execute("alter table hashes drop column index")

# extract hash from PostgreSQL
# engine.execute("SELECT * FROM hashes").fetchall()
