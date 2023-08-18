# from django.shortcuts import render
# from .sentiment import predict_sentiment
# from django.http import JsonResponse

# def analyze_sentiment(request):
#     if request.method == 'POST':
#         text = request.POST.get('text')
#         result = predict_sentiment(text)

#         if result == 'negative':
#             result = "Negative"
#         elif result == 'positive':
#             result = "Positive"
#         else:
#             result = "Neutral"

#         return render(request, 'sentiment_analysis.html', {'result': result})
#     return render(request, 'sentiment_analysis.html', {'result': None})


from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .sentiment import predict_sentiment

@csrf_exempt
def analyze_sentiment(request):
    if request.method == 'POST':
        text = request.POST.get('text')
        result = predict_sentiment(text)

        if result == 'negative':
            result = "Negative"
        elif result == 'positive':
            result = "Positive"
        else:
            result = "Neutral"

        return JsonResponse({'result': result})

    return JsonResponse({'result': None})
