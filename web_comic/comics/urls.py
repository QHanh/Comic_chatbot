from django.urls import path
from .views import story_list, chatbot_api, chatbot_page

urlpatterns = [
    path("", story_list, name="story_list"),
    path("api/chatbot/", chatbot_api, name="chatbot_api"),
    path("chatbot/", chatbot_page, name="chatbot"),
]
