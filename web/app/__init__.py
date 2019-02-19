import os
os.environ['PYTHONPATH']='python3'
from flask import Flask
app = Flask(__name__)
from app import views

