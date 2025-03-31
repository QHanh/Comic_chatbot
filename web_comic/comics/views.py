from django.shortcuts import render
from .models import Story
from comics.chatbot.chat_bot import main
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

def story_list(request):
    stories = Story.objects.exclude(image__isnull=True).exclude(image="")
    return render(request, "story_list.html", {"stories": stories})

@csrf_exempt
def chatbot_view(request):
    """
    API nhận query từ người dùng và trả về kết quả từ chatbot.
    """
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            query = data.get("query", "")

            if not query:
                return JsonResponse({"error": "Query không được để trống"}, status=400)

            answer = main(query)
            return JsonResponse({"query": query, "answer": answer})
        
        except json.JSONDecodeError:
            return JsonResponse({"error": "Dữ liệu không hợp lệ"}, status=400)

    return JsonResponse({"error": "Chỉ hỗ trợ phương thức POST"}, status=405)

def chatbot_page(request):
    return render(request, "chat_bot.html")


