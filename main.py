from flask import Flask, request, render_template
import pickle

app = Flask(__name__, template_folder='template')

# Load the saved model
with open('best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    age = float(request.form['age'])
    gender = float(request.form['gender'])
    bmi = float(request.form['bmi'])
    alcohol_consumption = float(request.form['alcohol_consumption'])
    smoking = float(request.form['smoking'])
    genetic_risk = float(request.form['genetic_risk'])
    physical_activity = float(request.form['physical_activity'])
    diabetes = float(request.form['diabetes'])
    hypertension = float(request.form['hypertension'])
    liver_function_test = float(request.form['liver_function_test'])

    features = [age, gender, bmi, alcohol_consumption, smoking, genetic_risk, physical_activity, diabetes, hypertension, liver_function_test]

    # Scale the input data
    scaled_features = scaler.transform([features])

    # Make the prediction
    prediction = best_model.predict(scaled_features)[0]

    if prediction == 0:
        result = "No Liver Disease"
    else:
        result = "Liver Disease"

    return render_template('result.html', result=result, age=age, gender=gender, bmi=bmi, alcohol_consumption=alcohol_consumption, smoking=smoking, genetic_risk=genetic_risk, physical_activity=physical_activity, diabetes=diabetes, hypertension=hypertension, liver_function_test=liver_function_test)

if __name__ == '__main__':
    app.run(debug=True)