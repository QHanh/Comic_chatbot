from django.urls import path
from .views import story_list, chatbot_view, chatbot_page

urlpatterns = [
    path("", story_list, name="story_list"),
    path("api/chatbot/", chatbot_view, name="chatbot_api"),
    path("chatbot/", chatbot_page, name="chatbot"),
]
