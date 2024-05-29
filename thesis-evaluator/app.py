from flask import Flask, render_template, abort
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import utils as ut
import pickle as pkl
import classifier as classifier
import feedback as feedback
import os
from flask_cors import CORS

load_dotenv()
ut.configure_logging()

#Initialise app
app = Flask(__name__)
CORS(app)

#Set the environment variables
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['UPLOAD_FOLDER'] = 'static/files/' #Change to input folder

class UploadFileForm(FlaskForm):
    file = FileField('File', validators=[InputRequired()])
    submit = SubmitField('Upload File')

#Load thesis and pass it through Logistic Regression classifier
def classify_grade(file_name):
    #Load classifier
    clf = pkl.load(open('classifier.pkl', 'rb'))
    #Get input thesis
    input_doc = classifier.prepare_test_doc('static/files/', file_name)
    grade = classifier.grade_doc(input_doc, clf)
    return grade

#Create vector database from thesis embeddings and give it to Mixtral
def prompt_mixtral():
    #Loading input thesis
    thesis = feedback.load_docs('static/files/')
    #Taking name of thesis for prompt later
    name = thesis[0]
    #Creating vector database with guidelines and input thesis
    pages = thesis
    feedback.create_vector_database(pages)
    #Creating retrieval chain and getting feedback from model
    k = 55
    retrieval_chain = feedback.create_chain(k)
    eval = ""
    check = False
    error = False
    i=0
    while "Conclusions" not in eval and not error and i < 3:
        check, eval = feedback.get_feedback(name, retrieval_chain, check)
        i = i+1
        if "The evaluator failed to process your document. Your text may be too long. Please make sure all appendices have been removed and re-upload it." in eval and k == 55:
            k = 25
            retrieval_chain = feedback.create_chain(k)
            continue
        elif "The evaluator failed to process your document. Your text may be too long. Please make sure all appendices have been removed and re-upload it." in eval or "An unexpected error occurred." in eval or "This document does not appear to be a thesis. Please make sure that you submitted the correct paper." in eval:
            error = True
        elif k==25:
            break
        else:
            error = False
    return error, eval

@app.route('/', methods=['GET', 'POST'])
def home():
    #Upload file
    form = UploadFileForm()
    if form.validate_on_submit():
        #Get the file
        file = form.file.data
        #Save the file
        file_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            app.config['UPLOAD_FOLDER'],
            secure_filename(file.filename))
        file.save(file_path)
        #Get feedback
        error, eval = prompt_mixtral()
        #Classify thesis
        if not error:
            grade = classify_grade(secure_filename(file.filename))
        else:
            grade = ""
        #Remove the file
        os.remove(file_path)
        return render_template('index.html',
                               form=form,
                               classification=grade,
                               evaluation=eval)
    return render_template('index.html', form=form)

@app.errorhandler(500)
def internal_server_error(error):
    return render_template('index.html', error="There was an issue with the server. Please refresh the page and try again in a few seconds.")

@app.errorhandler(502)
def unexpected_error(error):
    return render_template('index.html', error="The evaluator took too long to respond. Your file may be too big. Try remove all appendices from your thesis. Please refresh the page and try again.")

#Running app
if __name__ == "__main__":
    app.run(debug=True)