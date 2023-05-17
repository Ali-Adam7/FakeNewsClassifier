import pickle
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("model.pkl","rb"))
Fake = pd.read_csv('Fake.csv')
Real = pd.read_csv('True.csv')
Fake['Label']=0
Real['Label']=1

Dataset = Real.append(Fake, ignore_index=True)
Dataset= Dataset.fillna("")




TFIDF = TfidfVectorizer(stop_words = "english")
X_train_tf = TFIDF.fit_transform(Dataset['text'])

@app.route("/")
def Home():
        return render_template("index.html")
    
@app.route("/predict", methods = ["POST"])
def predict():
    
        data = request.form["news"]
        X_test_tf = TFIDF.transform([data])
        y_pred = model.predict(X_test_tf)
        result = "Fake"
        if(y_pred == 1):
            result = "Real"
        
        return render_template("index.html", prediction_text = "The news is predicted to be: {}".format(result))
if __name__ == '__main__':
    app.run()