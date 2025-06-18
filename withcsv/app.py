from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained model
with open('./withcsv/soc_classifier_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    Tannual = float(request.form['Tannual'])
    Pannual = float(request.form['Pannual'])
    Tillage = str(request.form['Tillage'])
    CoverCropGroup = str(request.form['CoverCropGroup'])
    GrainCropGroup = str(request.form['GrainCropGroup'])
    # Prepare the input for prediction
    if Tillage == 'Yes':
        Tillage = 1
    if CoverCropGroup == 'Yes':
        CoverCropGroup = 1
    user_input = np.array([[Tannual, Pannual, Tillage, CoverCropGroup, GrainCropGroup]])
    #  Make prediction
    # need to make soc_predictor into a python file to call x_train and y_train to fit the model
    model.fit(X_train, y_train)
    # change this to user_input once it works
    prediction = model.predict([22, 900, 1, 1, 0])
    return f'The predicted value is: {prediction[0]}'

if __name__ == "__main__":
    app.run(debug=True)
