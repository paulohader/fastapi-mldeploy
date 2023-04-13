

# 1. Import libraries
import uvicorn
from fastapi import FastAPI
from BankNotes import BankNote
import numpy as np
import pandas as pd
import pickle

# 2. Create the API
app = FastAPI()
pickle_in = open("classifier.pkl", "rb")
classifier=pickle.load(pickle_in)

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return{'message': 'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/NameHere
@app.get('/{name}')
def get_name(name: str):
    return {'message': f'Hello, {name}'}

# 5. Expose and make the prediction, JSON data and return the predicted Bank Mote
@app.post('/predcit')
def predict_banknote(data:BankNote):
    data = data.dict()
    variance=data['variance']
    skewness=data['skewness']
    curtosis=data['curtosis']
    entropy=data['entropy']
    # print(classifier.predict([[variance, skewness, curtosis, entropy]]))
    prediction = classifier.predict ([[variance, skewness, curtosis, entropy]])
    if(prediction[0]>0.5):
        prediction ="Fake note"
    else:
        prediction = "It's a Bank note"
    return {
        'prediction': prediction
    }
    

# 6. Run the API with uvicorn
#    It will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
# to run the app, type in the terminal (within the folder of the files):
# uvicorn app:app --reload