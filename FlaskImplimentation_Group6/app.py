import re
from flask import Flask, render_template, request
from flask.helpers import url_for
from werkzeug.utils import redirect
# import keras
# import tensorflow as tf
# from keras.models import load_model
# import pandas as pd
import flask

app = Flask(__name__)

# we need to redefine our metric function in order 
# to use it when loading the model 
def auc(y_true, y_pred):
    # auc = tf.metrics.auc(y_true, y_pred)[1]
    # keras.backend.get_session().run(tf.local_variables_initializer())
    return auc

global graph

# graph = tf.get_default_graph()

# model = load_model('NN_model5.h5', custom_objects={'auc': auc})


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods = ['POST','GET'])
def submit():
    if request.method=="POST":
        v1 = request.form['intaketermcode']
        v2 = request.form['intakecollegeexprience']
        v3 = request.form['schoolcode']
        programlongname = request.form['programlongname']
        progsem = request.form['progsem']
        studenrlvlname = request.form['studenrlvlname']
        timestatusname = request.form['timestatusname']
        residency = request.form['residency']
        funsrcname = request.form['funsrcname']
        mailcityname = request.form['mailcityname']
        mailpostcode3 = request.form['mailpostcode3']
        # mailpostcode = request.full_path['mailpostcode']
        mailprovname = request.form['mailprovname']
        gen = request.form['gen']
        dis = request.form['dis']
        mailcntname = request.form['mailcntname']
        staystatus = request.form['staystatus']
        termenrol = request.form['termenrol']
        acaper = request.form['acaper']
        exgrade = request.form['exgrade']
        suclvl = request.form['suclvl']
        percount = request.form['percount']
        agegrp = request.form['agegrp']
        catname = request.form['catname']
        tarsegname = request.form['tarsegname']
        edulvlname = request.form['edulvlname']
        
        features_to_keep = ["INTAKETERMCODE", "INTAKECOLLEGEEXPERIENCE", "SCHOOLCODE", "PROGRAMLONGNAME", "STUDENTLEVELNAME", "TIMESTATUSNAME", "RESIDENCYSTATUSNAME", "FUNDINGSOURCENAME", "MAILINGCITYNAME", "MAILINGPROVINCENAME", "GENDER", "DISABILITYIND", "MAILINGCOUNTRYNAME","CURRENTSTAYSTATUS", "FUTURETERMENROL", "ACADEMICPERFORMANCE", "EXPECTEDGRADTERMCODE", "FIRSTYEARPERSISTENCECOUNT", "AGEGROUPLONGNAME", "APPLICANTCATEGORYNAME", "PREVEDUCREDLEVELNAME"]

        # arr = pd.DataFrame([[v1, v2, v3, programlongname, progsem, studenrlvlname, timestatusname, residency, funsrcname, mailcityname, mailpostcode3, mailprovname, gen, dis, mailcntname, staystatus, termenrol, acaper, exgrade, suclvl, percount, agegrp, catname, tarsegname, edulvlname]], columns=features_to_keep)


        # prediction = model.predict(arr)
        return v1
        # return flask.jsonify(prediction)



if __name__ == "__main__":
    app.run(debug=True)