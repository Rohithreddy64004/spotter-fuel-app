from django.db import models


class TruckStop(models.Model):
    opis_id = models.IntegerField(db_index=True)
    name = models.CharField(max_length=255)
    address = models.CharField(max_length=255)
    city = models.CharField(max_length=100, db_index=True)
    state = models.CharField(max_length=10, db_index=True)
    retail_price = models.FloatField(db_index=True)
    latitude = models.FloatField(db_index=True)
    longitude = models.FloatField(db_index=True)

    class Meta:
        indexes = [
            models.Index(fields=['latitude', 'longitude']),
            models.Index(fields=['state', 'city']),
        ]

    def __str__(self):
        return f"{self.name} - {self.city}, {self.state} (${self.retail_price:.3f})"
