import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
 
# Load the dataset
heavy=pd.read_csv('complex_height_weight_data.csv')

heavy['Height(M)'] = heavy['Height'].str.extract(r'(\d+\.\d+|\d+)', expand=False).astype('float')
heavy['Weight(Kg)'] = heavy['Weight'].str.extract(r'(\d+\.\d+|\d+)', expand=False).astype('float')
heavy.drop(['Height','Weight'],axis=1,inplace=True)
heavy['Height(M)']=heavy['Height(M)'].fillna(heavy['Height(M)'].mean())
heavy['Weight(Kg)']=heavy['Weight(Kg)'].fillna(heavy['Weight(Kg)'].mean())
heavy['Height(M)']=heavy['Height(M)'].div(100)

 

 
# Split the dataset
X_train , X_test, y_train, y_test = train_test_split(heavy['Height(M)'].values.reshape(-1,1),heavy['Weight(Kg)'].values, random_state=11)
 
# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
 
# Save the trained model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as model_file:
	model = pickle.load(model_file)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	features = [float(x) for x in request.form.values()]
	final_features = np.array(features).reshape(1, -1)
	prediction = model.predict(final_features)
	return render_template('index.html', prediction_text='Your weight is : {:.2f} Kg'.format(prediction[0]))

if __name__ == '__main__':
	app.run(debug=False, host='0.0.0.0')
