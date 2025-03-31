from django.db import models

class Story(models.Model):
    title = models.CharField(max_length=255)
    url = models.URLField(default="unknown") 
    genres = models.TextField(blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    
    image = models.ImageField(upload_to="images/", blank=True, null=True)  
    image_url = models.URLField(blank=True, null=True)  

    def __str__(self):
        return self.title
