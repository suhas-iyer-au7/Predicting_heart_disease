import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


def ValuePredictor(to_predict_list): 
    to_predict = np.array(to_predict_list).reshape(1, 13) 
    result = model.predict(to_predict) 
    return result[0]
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    if int(output)== 1: 
            prediction ='High chances of heart disease'
    else: 
            prediction ='Chances of heart disease is less'            
    return render_template('index.html', prediction_text='{}'.format(prediction))




# @app.route('/result', methods = ['POST']) 
# def result(): 
#     if request.method == 'POST': 
#         to_predict_list = request.form.to_dict() 
#         to_predict_list = list(to_predict_list.values()) 
#         to_predict_list = list(map(int, to_predict_list)) 
#         result = ValuePredictor(to_predict_list)         
#         if int(result)== 1: 
#             prediction ='Income more than 50K'
#         else: 
#             prediction ='Income less that 50K'            
#         return render_template("result.html", prediction = prediction)

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)