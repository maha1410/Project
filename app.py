from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	df= pd.read_csv("Data/train_tweets.csv")
	df_data = df[["tweet","label"]]
	# Features and Labels
	df_x = df_data['tweet']
	df_y = df_data.label
    # Extract Feature With CountVectorizer
	corpus = df_x
	cv = CountVectorizer()
	#X = cv.fit_transform(corpus) # Fit the Data
	X = cv.fit_transform(df['tweet'].values.astype('U'))
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.20, random_state=42)
	#MLPClassifier
	from sklearn.neural_network import MLPClassifier
	clf = MLPClassifier(random_state=42)
	clf.fit(X_train,y_train)
	clf.score(X_test,y_test)
	#Alternative Usage of Saved Model
	# ytb_model = open("MLP_model.pkl","rb")
	# clf = joblib.load(ytb_model)

	if request.method == 'POST':
		comment = request.form['comment']
		data = [comment]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)
