from flask import Flask, render_template, request,jsonify
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from urllib.parse import quote as url_quote


app = Flask(__name__)
#app = Flask(__name__, static_folder='K:\\fake\\static', static_url_path='/static')
#app = Flask(__name__, static_folder='static', static_url_path='/static')

# Define stemming function 
port_stem = PorterStemmer()
def stemming(content):
    con = re.sub('[^a-zA-Z]', ' ', content)
    con = con.lower()
    con = con.split()
    con = [port_stem.stem(word) for word in con if not word in stopwords.words('english')]
    con = ' '.join(con)
    return con
# Download stopwords
nltk.download('stopwords')

# Load the model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vector.pkl')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('news')
    print("Received text:", text) # Use .get to avoid KeyError 
    if not text: 
        return jsonify({"error": "No input text provided"}), 400
    
    # Preprocess the text
    preprocessed_text = stemming(text)
    
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)
    
    result = "Fake News" if prediction[0] == 1 else "Real News"
    return jsonify(result=result)

if __name__ == "__main__":
    app.run(debug=True)

