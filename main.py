from flask import Flask, request, render_template, redirect, url_for, flash, json, jsonify
import numpy as np
import pandas as pd
import pickle

from flask_cors import CORS
from flask_pymongo import PyMongo
from bson import ObjectId
import datetime
import os

# flask app
app = Flask(__name__)

CORS(app)

app.secret_key = 'final'

# Configure MongoDB use root and password
app.config['MONGO_URI'] = 'mongodb://mongodb.default.svc.cluster.local/abc'

# Initialize PyMongo with the app
mongo = PyMongo(app)

# load database dataset
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/diets.csv")

# load model symptoms
svc = pickle.load(open("models/svc.pkl", "rb"))


# load model chatbot
# path_chatbot = "models/medical_chatbot"
# device = "cuda" if torch.cuda.is_available() else "cpu"
# tokenizer = GPT2Tokenizer.from_pretrained(path_chatbot)
# model = GPT2LMHeadModel.from_pretrained(path_chatbot, local_files_only=True).to(device)


def calculate_age(date_of_birth):
    # Calculate the age of a patient based on their date of birth
    today = datetime.date.today()
    age = today.year - date_of_birth.year
    # Subtract 1 from the age if the patient hasn't had their birthday this year
    if today < datetime.date(today.year, date_of_birth.month, date_of_birth.day):
        age -= 1
    return age


# custome and helping functions
# helper funtions
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis]['workout']

    return desc, pre, med, die, wrkout


symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4,
                 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9,
                 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13,
                 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18,
                 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22,
                 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27,
                 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
                 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37,
                 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42,
                 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46,
                 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50,
                 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54,
                 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58,
                 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61,
                 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66,
                 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70,
                 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74,
                 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78,
                 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82,
                 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86,
                 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89,
                 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92,
                 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96,
                 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100,
                 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103,
                 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107,
                 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110,
                 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113,
                 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116,
                 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120,
                 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124,
                 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127,
                 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction',
                 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma',
                 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)',
                 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A',
                 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis',
                 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)',
                 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism',
                 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis',
                 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection',
                 35: 'Psoriasis', 27: 'Impetigo'}


# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]


# creating routes
@app.route("/")
def index():
    return render_template("index.html")


# Define a route for the home page
@app.route('/symptoms', methods=['GET', 'POST'])
def symptom():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        # mysysms = request.form.get('mysysms')
        # print(mysysms)
        print(symptoms)
        if symptoms == "Symptoms":
            message = "Please either write symptoms or you have written misspelled symptoms"
            return render_template('symptoms.html', message=message)
        else:

            # Split the user's input into a list of symptoms (assuming they are comma-separated)
            user_symptoms = [s.strip() for s in symptoms.split(',')]
            # Remove any extra characters, if any
            user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
            predicted_disease = get_predicted_value(user_symptoms)
            dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

            my_precautions = []
            for i in precautions[0]:
                my_precautions.append(i)

            return render_template('symptoms.html', predicted_disease=predicted_disease, dis_des=dis_des,
                                   my_precautions=my_precautions, medications=medications, my_diet=rec_diet,
                                   workout=workout)

    return render_template('symptoms.html')


# chatbot
# @app.route('/chatbot', methods=['GET', 'POST'])
# def chatbot():
#     if request.method == 'POST':
#         user_input = request.form.get('user_input')
#         prompt_input = (
#             "The conversation between human and AI assistant.\n"
#             "[|Human|] {input}\n"
#             "[|AI|]"
#         )
#         sentence = prompt_input.format_map({'input': user_input})
#         inputs = tokenizer(sentence, return_tensors="pt").to(device)
# 
#         with torch.no_grad():
#             beam_output = model.generate(**inputs,
#                                          min_new_tokens=1,
#                                          max_length=512,
#                                          num_beams=3,
#                                          repetition_penalty=1.2,
#                                          early_stopping=True,
#                                          eos_token_id=198
#                                          )
#             full_response = tokenizer.decode(beam_output[0], skip_special_tokens=True)
#             # Extract only the answer part of the AI's response
#             response = full_response.split('[|AI|]')[-1].strip()
#             return render_template('chatbot.html', user_input=user_input, response=response)
# 
#     return render_template('chatbot.html')


# about view funtion and path
@app.route('/information')
def information():
    action = request.args.get('action')
    page = int(request.args.get('page', 1))  # Get the current page number from the URL, default is 1
    per_page = 50  # Number of patients per page

    if action == 'create':
        return render_template('information.html', action='create')
    elif action == 'list':
        # Fetch all patients from the data collection
        all_patients_cursor = mongo.db.data.find()

        all_patients = list(all_patients_cursor)

        gender_counts = {"M": 0, "F": 0}
        for patient in all_patients:
            gender_counts[patient["gender"]] += 1

        gender_counts_json = json.dumps(gender_counts)

        # Calculate the age of each patient
        ages = [calculate_age(patient["date_of_birth"]) for patient in all_patients]

        # Classify patients into age groups
        age_classification = {
            "0-10": 0,
            "11-20": 0,
            "21-30": 0,
            "31-40": 0,
            "41-50": 0,
            "51-60": 0,
            "61-70": 0,
            "71-80": 0,
            "81-90": 0,
            "91-100": 0
        }

        for age in ages:
            if age <= 10:
                age_classification["0-10"] += 1
            elif age <= 20:
                age_classification["11-20"] += 1
            elif age <= 30:
                age_classification["21-30"] += 1
            elif age <= 40:
                age_classification["31-40"] += 1
            elif age <= 50:
                age_classification["41-50"] += 1
            elif age <= 60:
                age_classification["51-60"] += 1
            elif age <= 70:
                age_classification["61-70"] += 1
            elif age <= 80:
                age_classification["71-80"] += 1
            elif age <= 90:
                age_classification["81-90"] += 1
            elif age <= 100:
                age_classification["91-100"] += 1

        age_classification_json = json.dumps(age_classification)
        print(age_classification_json)

        # Calculate the total number of patients and the total number of pages
        total_patients = mongo.db.data.count_documents({})
        total_pages = (total_patients + per_page - 1) // per_page

        # Fetch patients according to the current page and limit results
        patients = mongo.db.data.find().skip((page - 1) * per_page).limit(per_page)

        return render_template('information.html', action='list', patients=patients, total_pages=total_pages,
                               current_page=page, gender_counts=gender_counts_json,
                               age_classification=age_classification_json)
    elif action == 'search':
        return render_template('information.html', action='search')
    else:
        return render_template("information.html")


# Backend for get data gender and age for visualization return json
@app.route('/data', methods=['GET'])
def get_data():
    all_patients_cursor = mongo.db.data.find()

    all_patients = list(all_patients_cursor)

    gender_counts = {"M": 0, "F": 0, "Other": 0}
    for patient in all_patients:
        gender_counts[patient["gender"]] += 1

    gender_counts_json = json.dumps(gender_counts)

    # Calculate the age of each patient
    ages = [calculate_age(patient["date_of_birth"]) for patient in all_patients]

    # Classify patients into age groups
    age_classification = {
        "0-10": 0,
        "11-20": 0,
        "21-30": 0,
        "31-40": 0,
        "41-50": 0,
        "51-60": 0,
        "61-70": 0,
        "71-80": 0,
        "81-90": 0,
        "91-100": 0
    }

    for age in ages:
        if age <= 10:
            age_classification["0-10"] += 1
        elif age <= 20:
            age_classification["11-20"] += 1
        elif age <= 30:
            age_classification["21-30"] += 1
        elif age <= 40:
            age_classification["31-40"] += 1
        elif age <= 50:
            age_classification["41-50"] += 1
        elif age <= 60:
            age_classification["51-60"] += 1
        elif age <= 70:
            age_classification["61-70"] += 1
        elif age <= 80:
            age_classification["71-80"] += 1
        elif age <= 90:
            age_classification["81-90"] += 1
        elif age <= 100:
            age_classification["91-100"] += 1

    age_classification_json = json.dumps(age_classification)
    print(age_classification_json)

    return jsonify({
        'gender_counts': gender_counts,
        'age_classification': age_classification
    })


@app.route('/search', methods=['GET'])
def search_patient():
    # Get the search query from the URL parameters
    search_query = request.args.get('query')
    page = int(request.args.get('page', 1))  # Get the current page number, default is 1

    # Define the number of search results per page
    per_page = 50

    # Perform a case-insensitive search for patients in the data collection
    # Query the collection based on the search query in the 'name' field
    # Calculate the total number of search results
    total_patients = mongo.db.data.count_documents({
        'name': {'$regex': search_query, '$options': 'i'}
    })

    # Calculate the total number of pages
    total_pages = (total_patients + per_page - 1) // per_page

    # Fetch search results according to the current page and limit results
    results = mongo.db.data.find({
        'name': {'$regex': search_query, '$options': 'i'}
    }).skip((page - 1) * per_page).limit(per_page)

    # Render the information.html template with the search results and pagination details
    # Pass 'list' as the action to indicate we want to display the list of patients
    return render_template('information.html', action='list', patients=results, total_pages=total_pages,
                           current_page=page)


# Create a new patient
@app.route('/create', methods=['GET', 'POST'])
def create_patient():
    if request.method == 'POST':
        # Get form data
        name = request.form.get('name')
        date_of_birth = request.form.get('date_of_birth')
        gender = request.form.get('gender')
        medical_conditions = request.form.get('medical_conditions')
        medications = request.form.get('medications')
        allergies = request.form.get('allergies')
        last_appointment_date = request.form.get('last_appointment_date')

        # Convert date_of_birth and last_appointment_date to datetime objects
        date_of_birth = datetime.datetime.strptime(date_of_birth, '%Y-%m-%d')
        last_appointment_date = datetime.datetime.strptime(last_appointment_date, '%Y-%m-%d')

        # Create a new patient record
        new_patient = {
            'name': name,
            'date_of_birth': date_of_birth,
            'gender': gender,
            'medical_conditions': medical_conditions,
            'medications': medications,
            'allergies': allergies,
            'last_appointment_date': last_appointment_date
        }

        # Insert the new patient record into the data collection
        mongo.db.data.insert_one(new_patient)

        # Redirect to the patient list page
        flash('Patient added successfully.')
        return redirect(url_for('information', action='list'))

    return render_template('create_patient.html')


# Edit a patient
@app.route('/edit/<_id>', methods=['GET', 'POST'])
def edit_patient(_id):
    # Query the patient from the database
    patient = mongo.db.data.find_one({'_id': ObjectId(_id)})

    if request.method == 'POST':
        # Get form data
        name = request.form.get('name')
        date_of_birth = request.form.get('date_of_birth')
        gender = request.form.get('gender')
        medical_conditions = request.form.get('medical_conditions')
        medications = request.form.get('medications')
        allergies = request.form.get('allergies')
        last_appointment_date = request.form.get('last_appointment_date')

        # Convert date_of_birth and last_appointment_date to datetime objects
        date_of_birth = datetime.datetime.strptime(date_of_birth, '%Y-%m-%d')
        last_appointment_date = datetime.datetime.strptime(last_appointment_date, '%Y-%m-%d')

        # Update the patient record
        mongo.db.data.update_one(
            {'_id': ObjectId(_id)},
            {'$set': {
                'name': name,
                'date_of_birth': date_of_birth,
                'gender': gender,
                'medical_conditions': medical_conditions,
                'medications': medications,
                'allergies': allergies,
                'last_appointment_date': last_appointment_date
            }}
        )

        # Redirect to the patient list page
        flash('Patient updated successfully.')
        return redirect(url_for('information', action='list'))

    # Convert dates for display in edit form
    if patient:
        patient['date_of_birth'] = patient['date_of_birth'].strftime('%Y-%m-%d')
        patient['last_appointment_date'] = patient['last_appointment_date'].strftime('%Y-%m-%d')
    return render_template('edit_patient.html', patient=patient)


# Delete a patient
@app.route('/delete/<_id>')
def delete_patient(_id):
    # Query the patient from the database
    patient = mongo.db.data.find_one({'_id': ObjectId(_id)})

    return render_template('delete_confirmation.html', patient=patient)


@app.route('/confirm_delete/<_id>', methods=['POST'])
def confirm_delete_patient(_id):
    # Delete the patient record from the data collection
    mongo.db.data.delete_one({'_id': ObjectId(_id)})

    # Redirect to the patient list page
    flash('Patient deleted successfully.')
    return redirect(url_for('information', action='list'))


if __name__ == '__main__':
    app.run(port=5000)
