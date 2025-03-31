import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web_comic.settings")
django.setup()

from comics.models import Story

IMAGE_DIR = "comics/static/images/"

image_files = sorted(
    [f for f in os.listdir(IMAGE_DIR) if f.endswith((".jpg", ".png"))],
    key=lambda x: os.path.getmtime(os.path.join(IMAGE_DIR, x))
)

stories = list(Story.objects.order_by("id"))

for i in range(min(len(image_files), len(stories))):
    story = stories[i]
    image_filename = image_files[i]
    
    story.image = f"images/{image_filename}" 
    story.save()

    print(f"Gán ảnh {image_filename} vào truyện {story.title}")
