from flask import Flask, render_template, request, jsonify
from urllib.parse import urlparse, parse_qs
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import string
import math
import pickle
from sklearn.ensemble import RandomForestClassifier
import sqlite3

app = Flask(__name__, template_folder='templates')

def init_db():
    with sqlite3.connect('scans.db') as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS scans 
                     (id INTEGER PRIMARY KEY, url TEXT, prediction TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()

init_db()

# Load the machine learning model
# data = pickle.load(open('random_forest_model.pkl', 'rb'))
# async def loading() :
# print(data)
#     return model
# Define URL feature extraction functions here
def extract_tld(url):
    """
    Extracts the top-level domain (TLD) from the URL.
    """
    parsed = urlparse(url)
    domain = parsed.netloc.split('.')[-1]
    return len(domain)

# Define other URL feature extraction functions here
def url_length(url):
    """
    Returns the length of the URL.
    """
    return len(url)

def domain_length(url):
    """
    Returns the length of the domain in the URL.
    """
    parsed = urlparse(url)
    domain = parsed.netloc.split('.')[0]
    return len(domain)

def filename_length(url):
    """
    Returns the length of the filename in the URL.
    """
    parsed = urlparse(url)
    return len(os.path.basename(parsed.path))

def path_url_ratio(url):
    """
    Calculates the ratio of path length to URL length.
    """
    parsed = urlparse(url)
    path_length = len(parsed.path)
    url_length = len(url)
    return path_length / url_length if url_length > 0 else 0

def num_dots_url(url):
    """
    Calculates the number of dots in URL 
    """
    count1 = url.count('.')
    return count1

def count_digits_query_string(url):
    """
    Counts the number of digits in the query string of the URL.
    """
    parsed_url = urlparse(url)
    query_string = parsed_url.query
    parsed_query = parse_qs(query_string)
    digit_count = sum(1 for value in parsed_query.values() for item in value if item.isdigit())
    return digit_count

def longest_path_token_length(url):
    """
    Returns the length of the longest token in the path of the URL.
    """
    parsed = urlparse(url)
    path_tokens = parsed.path.split('/')
    return max(len(token) for token in path_tokens)

def count_delimiters_domain(url):
    """
    Counts the number of delimiters in the domain of the URL.
    """
    parsed = urlparse(url)
    domain = parsed.netloc.split('.')[0]
    return sum(1 for char in domain if char in string.punctuation)

def count_delimiters_path(url):
    """
    Counts the number of delimiters in the path of the URL.
    """
    parsed = urlparse(url)
    return sum(1 for char in parsed.path if char in string.punctuation)   

def symbol_count_domain(url):
    """
    Counts the number of symbols in the domain of the URL.
    """
    parsed = urlparse(url)
    domain = parsed.netloc.split('.')[0]
    return sum(1 for char in domain if char in string.punctuation)

def entropy(s):
    """
    Calculate the entropy of a given string.
    """
    probabilities = [float(s.count(c)) / len(s) for c in set(s)]
    entropy = - sum(p * math.log(p) / math.log(2.0) for p in probabilities)
    return entropy

def url_entropy(url):
    """
    Calculate the entropy of a URL.
    """
    # Removing protocol and www if present
    if url.startswith("http://"):
        url = url[len("http://"):]
    elif url.startswith("https://"):
        url = url[len("https://"):]
    if url.startswith("www."):
        url = url[len("www."):]
        
    # Removing special characters and spliting into characters
    url = ''.join(e for e in url if e.isalnum())
    
    # Calculating entropy
    return entropy(url)

def entropy_domain(url):  
    """
    Calculates the entropy of the domain name in the URL.
    """
    parsed = urlparse(url)
    domain = parsed.netloc.split('.')[0]
    length = len(domain)
    if length <= 1:
        return 0
    else:
        entropy = 0
        for char in string.ascii_lowercase:
            p_i = domain.count(char) / length
            if p_i > 0:
                entropy -= p_i * math.log2(p_i)
        return entropy

def count_hyphen(url):
    """
    Count the occurrences of hyphen (-) in the input string.
    """
    return url.count('-')

def count_slash(url):
    """
    Count the occurrences of slash (/) in the input string.
    """
    return url.count('/')

def count_question_mark(url):
    """
    Count the occurrences of question mark (?) in the input string.
    """
    return url.count('?')

def count_equal(url):
    """
    Count the occurrences of equal sign (=) in the input string.
    """
    return url.count('=')

def count_at(url):
    """
    Count the occurrences of at sign (@) in the input string.
    """
    return url.count('@')

def count_exclamation(url):
    """
    Count the occurrences of exclamation mark (!) in the input string.
    """
    return url.count('!')

def count_tilde(url):
    """
    Count the occurrences of tilde (~) in the input string.
    """
    return url.count('~')

def count_comma(url):
    """
    Count the occurrences of comma (,) in the input string.
    """
    return url.count(',')

def count_plus(url):
    """
    Count the occurrences of plus sign (+) in the input string.
    """
    return url.count('+')

def count_star(url):
    """
    Count the occurrences of asterisk (*) in the input string.
    """
    return url.count('*')

def count_hash(url):
    """
    Count the occurrences of hashtag (#) in the input string.
    """
    return url.count('#')

def count_dollar(url):
    """
    Count the occurrences of dollar sign ($) in the input string.
    """
    return url.count('$')


def test_it(url):
    """
    Control
    Returns a list with features
    """
    features = []

    features.append(extract_tld(url))
    features.append(url_length(url))
    features.append(domain_length(url))
    features.append(filename_length(url))
    features.append(path_url_ratio(url))
    features.append(num_dots_url(url))
    features.append(count_digits_query_string(url))
    features.append(longest_path_token_length(url))
    features.append(count_delimiters_domain(url))
    features.append(count_delimiters_path(url))
    features.append(symbol_count_domain(url))
    features.append(url_entropy(url))
    features.append(entropy_domain(url))
    features.append(count_hyphen(url))
    features.append(count_slash(url))
    features.append(count_question_mark(url))
    features.append(count_equal(url))
    features.append(count_at(url))
    features.append(count_exclamation(url))
    features.append(count_tilde(url))
    features.append(count_comma(url))
    features.append(count_plus(url))
    features.append(count_star(url))
    features.append(count_hash(url))
    features.append(count_dollar(url))

    return features
@app.route('/')
def home():
    return render_template('index.html')

data = pd.read_csv('enhanced_feature_set.csv')
print("Training model on startup...")
model = RandomForestClassifier()
X = data.drop(["class"], axis=1)
y = data["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
print("Model trained successfully!")

@app.route('/analyze', methods=['POST'])
def predict():
    if 'url' in request.form:
        url = request.form['url'].strip()
        
        # Ensure the URL has a scheme so urlparse works correctly
        if not url.startswith('http://') and not url.startswith('https://'):
            url = 'http://' + url
            
        print(f"Received URL: {url}")
        # Extract features from the URL
        try:
            features = test_it(url)
        except Exception as e:
            return jsonify({'error': str(e)}), 400
            
        xnew = [features]
        ynew = model.predict(xnew)
        prediction = ynew[0]
        
        # Generate Explainable AI Report (User-Friendly)
        report = []
        if prediction == 'defacement' or prediction != 'benign':
            if features[1] > 75: 
                report.append(f"Unusually Long Link: The web address has {features[1]} characters. Hackers often use extremely long links to hide dangerous content at the very end.")
            if features[5] > 3: 
                report.append(f"Too Many Dots: We found {features[5]} dots in the link. This is a common trick used to create fake website names that look like real ones.")
            if features[6] > 20: 
                report.append(f"Suspicious Numbers: There are {features[6]} numbers hidden in the link. This usually means an automated script or tracker is trying to sneak through.")
            if features[7] > 40: 
                report.append(f"Hidden Folders: A section of the link is abnormally long ({features[7]} characters). Normal websites rarely use folders this long.")
            if features[11] > 4.5: 
                report.append(f"Scrambled Text: The link looks highly randomized or scrambled. Hackers do this to bypass security filters and hide their true destination.")
            if features[12] > 3.5: 
                report.append(f"Fake Domain Name: The main website name looks like random gibberish. This is a big red flag that the site was generated automatically by a bot.")
            if features[16] > 3: 
                report.append(f"Too Many Commands: We found {features[16]} '=' signs. This means the link is trying to send a lot of hidden commands to the website, which is a common attack method.")
            if features[17] > 0: 
                report.append(f"Fake Destination Trick: The link contains an '@' symbol. This is an old trick used to lie to you about what website you are actually visiting.")
            if features[9] > 5:
                report.append(f"Suspicious Symbols: There are a lot of symbols ({features[9]}) in the web address. Attackers use these to try and break into locked parts of a server.")
                
            if len(report) == 0: 
                report.append("Hidden Threat: Our AI model analyzed the overall structure of the link and found hidden patterns commonly used by hackers.")
        else:
            report.append("Safe Link: The length, structure, and symbols in this web address all look completely normal and safe.")
            
        # Log to Database
        try:
            with sqlite3.connect('scans.db') as conn:
                c = conn.cursor()
                c.execute("INSERT INTO scans (url, prediction) VALUES (?, ?)", (url, prediction))
                conn.commit()
        except Exception as e:
            print("DB Error:", e)

        return jsonify({'url': url, 'prediction': prediction, 'report': report})
    else:
        return jsonify({'error': 'URL key not found in form data'})

@app.route('/history', methods=['GET'])
def history():
    try:
        with sqlite3.connect('scans.db') as conn:
            c = conn.cursor()
            c.execute("SELECT url, prediction, timestamp FROM scans ORDER BY id DESC LIMIT 10")
            rows = c.fetchall()
            return jsonify([{'url': r[0], 'prediction': r[1], 'timestamp': r[2]} for r in rows])
    except Exception as e:
        return jsonify([])
    # return "hallo"

if __name__ == '__main__':
    app.run(debug=True,host = '0.0.0.0')
