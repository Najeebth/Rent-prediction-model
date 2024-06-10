from django.shortcuts import render
import numpy as np
import pickle
from django.http import JsonResponse
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response

# Load your model
model1 = pickle.load(open('predictor/decision_tree_model.pkl', 'rb'))

def home(request):
    return render(request, 'index.html')

@api_view(['POST'])
def predict(request):
    if request.method == 'POST':
        int_features = [int(x) for x in request.data.values()]
        final_features = [np.array(int_features)]
        prediction = model1.predict(final_features)

        output = round(prediction[0], 2)

        return render(request, 'index.html', {'prediction_text': 'Rent should be Rs. {}'.format(output)})

@api_view(['POST'])
def predict_api(request):
    if request.method == 'POST':
        data = request.data
        prediction = model1.predict([np.array(list(data.values()))])

        output = prediction[0]
        return Response(output)


