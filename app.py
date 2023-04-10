from flask import Flask, render_template, request, redirect, url_for
import os
import streamlit as st

app = Flask(__name__)  # create the application instance :)
app.config.from_object(__name__)  # load config from this file , flaskr.py

# Load default config and override config from an environment variable
app.config.update(dict(
     DATABASE=os.path.join(app.root_path, 'flaskr.db'),
))
app.config.from_envvar('FLASKR_SETTINGS', silent=True)

@app.route('/')  # at the end point /
def index():  # call method hello
     st.write("hello world")
     
     return render_template('index.html')  # which returns "hello world"




