from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
with open('./soc_classifier_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

#display html page
@app.route('/')
def home():
    return render_template('index.html')

# get input and make the prediction on another html page
@app.route('/predict', methods=['GET','POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        # Get user input from the form
        Tannual = float(request.form['Tannual'])
        Pannual = float(request.form['Pannual'])
        Tillage = str(request.form['Tillage'])
        CoverCropGroup = str(request.form['CoverCropGroup'])
        GrainCropGroup = str(request.form['GrainCropGroup'])

        # Create a DataFrame for the input
        user_input = pd.DataFrame({
            'Tannual': [Tannual],
            'Pannual': [Pannual],
            'Tillage': [Tillage],
            'CoverCropGroup': [CoverCropGroup],
            'GrainCropGroup': [GrainCropGroup]
        })

        #  Make prediction
        prediction = model.predict(user_input)

    return render_template('prediction.html', prediction=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
