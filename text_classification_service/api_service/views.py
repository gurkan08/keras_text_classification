

from rest_framework.views import APIView
from rest_framework.response import Response
from .codes.api_ import api

# Create your views here.

class TextClassificationView(APIView):

    def get(self, request):
        text = request.data["text"]
        content = {"result": api(text)}
        return Response(content)

    def post(self, request):
        text = request.data["text"]
        content = {"result": api(text)}
        return Response(content)

