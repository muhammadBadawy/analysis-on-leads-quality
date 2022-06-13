import pyghaseel.Cleaner
from meaningless import *
import json
import pickle
from flask import Flask, Response, request
import re

app = Flask(__name__)

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

filename = 'finalized_vectiorizer.sav'
loaded_vectorizer = pickle.load(open(filename, 'rb'))

threshold = 0.5

cleaner = pyghaseel.Cleaner()

def text_clean(text):
    return cleaner.clean(text, ar=True, en=True)

def removeMeaningless(text):
    for word in meaningless_words:
        text = text.replace(word, "")
    return text

removeSpace = lambda x:str(re.sub(' +', ' ', x))
removeN = lambda x:str(re.sub(r'\\n', ' ', x))

def removeMessagesMeaningless(text):
    for word in messages_meaningless_words:
        text = text.replace(word, "")
    return text



# columns = [
#     "lead_mobile_network",
#     "method_of_contact",
#     "ad_group",
#     "lead_source",
#     "campaign",
#     "location",
#     "message"
# ]


@app.route('/isGoodLead', methods=['POST'])
def isGoodLead():
    my_input = ""
    
    for field in request.form.values():
        field = removeN(field)
        field = removeSpace(field)
        field = text_clean(field)
        field = removeMeaningless(field)
        my_input += field+" "
    
    my_input.strip()

    print(my_input)
    
    my_input = loaded_vectorizer.transform([my_input])[0]
    
    result = loaded_model.predict(my_input)[0]
    print(loaded_model.predict(my_input))
    print(loaded_model.predict_proba(my_input))
    
#     result = {"is_good_lead":True if result >= threshold else False}
    result = {"is_good_lead":result}
    

    return Response(
            response=json.dumps(result, ensure_ascii=False).encode('utf8'),
            status=200,
            mimetype='application/json'
        )


if __name__ == '__main__':
    app.run(debug = True, port = 5200, threaded=False, processes=0)
