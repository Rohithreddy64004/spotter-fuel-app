import os
import csv
import json
import time
from pathlib import Path
from django.core.management.base import BaseCommand
from fuel_route.models import TruckStop


class Command(BaseCommand):
    help = "Loads fuel prices from a CSV file into the database with pre-populated geocoding."

    def add_arguments(self, parser):
        parser.add_argument(
            "csv_file",
            type=str,
            nargs="?",
            default="fuel-prices-for-be-assessment.csv",
            help="Path to the fuel prices CSV file",
        )
        parser.add_argument(
            "--clear",
            action="store_true",
            help="Clear existing TruckStop records before importing",
        )

    def handle(self, *args, **options):
        csv_file_path = options["csv_file"]
        clear = options["clear"]

        if not os.path.exists(csv_file_path):
            self.stderr.write(self.style.ERROR(f"CSV file not found: {csv_file_path}"))
            return

        # Load pre-populated offline city & state coordinates
        coords_path = Path(__file__).resolve().parent.parent.parent / "data" / "city_coords.json"
        if not coords_path.exists():
            self.stderr.write(self.style.ERROR(f"Coordinate dictionary missing at {coords_path}"))
            return

        with open(coords_path, "r", encoding="utf-8") as f:
            geo_data = json.load(f)

        cities_db = geo_data.get("cities", {})
        states_db = geo_data.get("states", {})

        self.stdout.write(f"Loaded {len(cities_db)} offline city coordinates.")

        if clear or TruckStop.objects.exists():
            deleted_count, _ = TruckStop.objects.all().delete()
            self.stdout.write(f"Cleared {deleted_count} previous TruckStop records.")

        start_time = time.time()
        truck_stops = []
        skipped = 0

        with open(csv_file_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    opis_id = int(row.get("OPIS Truckstop ID", 0).strip())
                    name = row.get("Truckstop Name", "").strip()
                    address = row.get("Address", "").strip()
                    city = row.get("City", "").strip()
                    state = row.get("State", "").strip().upper()
                    price_str = row.get("Retail Price", "0").strip()
                    retail_price = float(price_str) if price_str else 0.0

                    city_key = f"{city.lower()}|{state}"
                    if city_key in cities_db:
                        lat, lon = cities_db[city_key]
                    elif state in states_db:
                        lat, lon = states_db[state]
                    else:
                        # US geographic centroid default
                        lat, lon = 39.8283, -98.5795

                    truck_stops.append(
                        TruckStop(
                            opis_id=opis_id,
                            name=name,
                            address=address,
                            city=city,
                            state=state,
                            retail_price=retail_price,
                            latitude=lat,
                            longitude=lon,
                        )
                    )
                except Exception as e:
                    skipped += 1

        # Bulk insert
        TruckStop.objects.bulk_create(truck_stops, batch_size=2000)
        elapsed = time.time() - start_time

        self.stdout.write(
            self.style.SUCCESS(
                f"Successfully ingested {len(truck_stops)} truck stops in {elapsed:.2f}s (Skipped: {skipped})."
            )
        )

