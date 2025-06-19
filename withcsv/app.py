from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load your trained model
with open('./withcsv/soc_classifier_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    prediction = None
    # user_input = {
    #     'tannual': '',
    #     'pannual': '',
    #     'tillage': 'None',  # Default value
    #     'cover_crop': 'None',  # Default value
    #     'grain_crop': 'Monoculture'  # Default value
    # }
    if request.method == 'POST':
        # Get user input from the form
        Tannual = float(request.form['Tannual'])
        Pannual = float(request.form['Pannual'])
        Tillage = str(request.form['Tillage'])
        CoverCropGroup = str(request.form['CoverCropGroup'])
        GrainCropGroup = str(request.form['GrainCropGroup'])

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
