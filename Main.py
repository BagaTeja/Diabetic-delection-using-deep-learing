from flask import Flask, render_template, request
from flask_material import Material
from Predict import Predict_Input as pi
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os

app = Flask(__name__)
Material(app)


@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")


@app.route('/retino')
def retino():
    return render_template("retino.html")


@app.route('/rpredict')
def rpredict():
    root = Tk()
    file = askopenfilename()
    if file:
        filename= os.path.abspath(file)
    root.withdraw()
    x = pi(filename)
    print("Predicted Value is ", x)
    return render_template("index.html")





if __name__ == '__main__':
	app.run(host='127.0.0.1', port='5001',debug=True)
