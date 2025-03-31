import os
import django
import pandas as pd

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web_comic.settings")
django.setup()

from comics.models import Story

df = pd.read_csv("comics/crawl/stories_updated.csv")

for index, row in df.iterrows():
    Story.objects.get_or_create(
        url=row["link truyện"],  
        defaults={
            "title": row["tiêu đề"],
            "genres": row["thể loại"],
            "description": row["mô tả"],
            "image_url": row["link ảnh"],
        }
    )
print("✅ Hoàn tất nhập dữ liệu!")
