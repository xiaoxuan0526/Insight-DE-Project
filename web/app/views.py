from app import app
from flask import Flask, render_template, request, redirect, url_for, make_response,jsonify
from werkzeug import secure_filename
import os
import cv2
from datetime import timedelta
import numpy as np
import pandas as pd
import random

os.environ['PYTHONPATH']='python3'

#allowable file format
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app.send_file_max_age_default = timedelta(seconds=1)

@app.route('/', methods=['POST', 'GET'])
def upload():
    return render_template('upload.html')

@app.route('/upload_ok', methods=['POST', 'GET'])
def upload_ok():
    if request.method == 'POST':
        f = request.files['file']
        if not (f and allowed_file(f.filename)):
            return jsonify({"error":1001, "msg":"only accepting .png, .PNG, .jpg, .JPG, jpeg or .bmp files"})
        user_input = request.form.get("name")
        basepath = os.path.dirname(__file__) #current directory
        basepath = os.path.join(basepath,'static/images')
        ID = str(random.randint(1,100000000))
        user_dir = '/home/ubuntu/app/static/images/'+ID
        createdir ='mkdir '+user_dir
        os.system(createdir)
        upload_path = os.path.join(user_dir, secure_filename(f.filename))
        f.save(upload_path)
        img = cv2.imread(upload_path)
        test1_save_path = os.path.join(user_dir, 'test1.jpg')
        cv2.imwrite(test1_save_path, img)
        os.system('rm '+user_dir+'/result1.jpg')
        os.system('rm '+user_dir+'/result2.jpg')
        os.system('rm '+user_dir+'/result3.jpg')
        os.system('rm '+user_dir+'/result4.jpg')
        os.system('rm '+user_dir+'/result5.jpg')
        os.system('rm '+user_dir+'/result6.jpg')
        os.system('export PYSPARK_PYTHON=python3')
        back_end = 'spark-submit --master local[*] --executor-memory 6G --py-files /home/ubuntu/IncomingImageDetection.py /home/ubuntu/IncomingImageDetection.py ' +user_dir+'/ 1000'
        os.system(back_end)
        return render_template('upload_ok.html',userinput=ID)

@app.route('/selection_submitted', methods=['POST', 'GET'])
def return_to_upload():
    if request.method == 'POST':
        print('result 1 user selection:',request.form.get('r1check'))
        print('result 2 user selection:',request.form.get('r2check'))
        print('result 3 user selection:',request.form.get('r3check'))
        print('result 4 user selection:',request.form.get('r4check'))
        print('result 5 user selection:',request.form.get('r5check'))
        print('result 6 user selection:',request.form.get('r6check'))
        return render_template('selection_submitted.html')
