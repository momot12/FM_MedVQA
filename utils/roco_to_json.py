import csv
import json
import random

def rocov_csv_to_json():
    js = json.load(open('sample_conversation.json', 'r'))
    result = []
    with open('sample_roco.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            randomQuestion = random.choice(js)['conversations'][0]['value'] # take random question
            jeson = {"image": row['ID'], "conversations": [{"from": "human", "value": randomQuestion}, {"from": "gpt", "value": row['Caption']}]}
            result.append(jeson)
    print (json.dumps(result, indent=1, separators=(',', ': ')))
                
rocov_csv_to_json()
            
