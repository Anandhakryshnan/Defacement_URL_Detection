from flask import Flask, render_template, request, jsonify
from urllib.parse import urlparse, parse_qs
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import string
import math
import pickle
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__, template_folder='templates')

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

@app.route('/analyze', methods=['POST'])
def predict():
    model = RandomForestClassifier()
    X = data.drop(["class"], axis=1)
    y = data["class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    if 'url' in request.form:
        url = request.form['url']
        print(f"Received URL: {url}")
        # Extract features from the URL
        features = test_it(url)
        # Make prediction using the loaded model
        print(features)
        xnew = [features]
        ynew = model.predict(xnew)

        print (ynew)
        # prediction = model.predict([features])  # Remove [0] indexing
        # Return the URL and prediction in the JSON response
        return jsonify({'url': url, 'prediction': ynew[0]})  # Adjust indexing here as well
    else:
        return jsonify({'error': 'URL key not found in form data'})
    # return "hallo"

if __name__ == '__main__':
    app.run(debug=True)
