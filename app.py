from flask import Flask, render_template, request
import numpy as np
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    bp = float(request.form['bp'])
    sg = float(request.form['sg'])
    al = float(request.form['al'])
    su = float(request.form['su'])
    bgr = float(request.form['bgr'])
    bu = float(request.form['bu'])
    sc = float(request.form['sc'])
    sod = float(request.form['sod'])
    pot = float(request.form['pot'])
    hemo = float(request.form['hemo'])
    rc = float(request.form['rc'])
    # htn = float(request.form['htn'])
    if request.form['htn']=="yes":
        htn = float(1)
    else: htn = float(0)
    #dm = float(request.form['dm'])
    if request.form['dm']=="yes":
        dm = float(3)
    else: dm = float(2)
    #appet = float(request.form['appet'])
    if request.form['appet']=="good":
        appet = float(0)
    else: appet = float(1)
    #pe = float(request.form['pe'])
    if request.form['pe']=="yes":
        pe = float(1)
    else: pe = float(0)

    #float_features = [float(x) for x in request.form.values()]
    features = np.array([[bp, sg, al, su, bgr, bu, sc, sod, pot, hemo, rc, htn, dm, appet, pe]])
    prediction = model.predict(features)
    return render_template('result.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)

