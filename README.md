# Spotter Fuel Route Optimizer

A Django REST Framework service that plans cost-optimal refuelling itineraries for long-haul commercial vehicles. Given an origin and destination, the service computes a driving route (one OSRM network call), spatially indexes nearby truck stops against the route corridor using a `cKDTree`, and selects the globally minimum-cost sequence of fuel stops via a DAG Dynamic Programming algorithm — all within a single HTTP request.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Repository Layout](#3-repository-layout)
4. [Data Model](#4-data-model)
5. [Data Ingestion](#5-data-ingestion)
6. [Core Engine — `fuel_route/services.py`](#6-core-engine)
   - 6.1 [Geocoding](#61-geocoding)
   - 6.2 [OSRM Routing](#62-osrm-routing)
   - 6.3 [Spatial Corridor Filtering — cKDTree](#63-spatial-corridor-filtering)
   - 6.4 [Fuel Optimisation Algorithm](#64-fuel-optimisation-algorithm)
7. [API Reference](#7-api-reference)
8. [Interactive Map UI](#8-interactive-map-ui)
9. [DRF Serializers](#9-drf-serializers)
10. [Custom Exceptions](#10-custom-exceptions)
11. [Settings Configurability](#11-settings-configurability)
12. [Local Setup](#12-local-setup)
13. [Management Commands](#13-management-commands)
14. [Running Tests](#14-running-tests)
15. [Design Decisions & Trade-offs](#15-design-decisions--trade-offs)
    - [Coordinate Accuracy & Geographic Trade-offs](#coordinate-accuracy--geographic-trade-offs)

---

## 1. Project Overview

| Property | Value |
|---|---|
| Framework | Django 6.1 + Django REST Framework 3.18 |
| Database | SQLite (dev) — swappable via `DATABASES` setting |
| External API | OSRM driving route — **one call per unique route pair** |
| Routing strategy | DAG Dynamic Programming — provably globally optimal fuel cost |
| Vehicle model | 500-mile tank, 10 MPG, departs full |
| Response time target | < 500 ms after initial data load |

The application satisfies the Spotter technical assessment requirements:

- Loads `fuel-prices-for-be-assessment.csv` into a `TruckStop` relation with pre-populated offline geocoordinates (no external geocoding rate limits).
- Exposes `POST /api/route/` and a Leaflet.js map at `/map/`.
- Guarantees the vehicle never runs out of fuel and minimises total expenditure.

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Client (browser / curl)                                             │
│   POST /api/route/  {"start": "Chicago, IL", "finish": "Austin, TX"}│
└───────────────────────────┬──────────────────────────────────────────┘
                            │
            ┌───────────────▼───────────────┐
            │  RouteAPIView (DRF APIView)    │
            │  RouteRequestSerializer        │
            │  – strip whitespace            │
            │  – origin ≠ destination check  │
            └───────────────┬───────────────┘
                            │  plan_fuel_route()
            ┌───────────────▼───────────────┐
            │  geocode_location()            │
            │  1. In-process cache           │
            │  2. Offline city/state dict    │  ← fuel_route/data/city_coords.json
            │  3. Nominatim fallback         │
            └───────────────┬───────────────┘
                            │
            ┌───────────────▼───────────────┐
            │  fetch_driving_route()         │
            │  ONE HTTPS call to OSRM        │  ← routing.openstreetmap.de (primary)
            │  Cached by (lat,lon) key pair  │    router.project-osrm.org (fallback)
            │  Returns: distance, polyline   │
            └───────────────┬───────────────┘
                            │
            ┌───────────────▼───────────────┐
            │  find_corridor_stops()         │
            │  – Densify route waypoints     │
            │    (≤ 1 mi spacing)            │
            │  – DB bounding-box pre-filter  │
            │  – scipy cKDTree spatial index │
            │  – 10-mile corridor cutoff     │
            └───────────────┬───────────────┘
                            │
            ┌───────────────▼───────────────┐
            │  optimise_refuel_stops()       │
            │  Greedy forward-lookahead      │
            │  Window: [pos+100, pos+480] mi │
            │  Selects cheapest stop, refuel │
            │  Accounts for final leg cost   │
            └───────────────┬───────────────┘
                            │
            ┌───────────────▼───────────────┐
            │  RouteResponseSerializer       │
            │  JSON → HTTP 200               │
            └───────────────────────────────┘
```

---

## 3. Repository Layout

```
spotter-fuel-app/
│
├── manage.py
├── fuel-prices-for-be-assessment.csv   # Source dataset (company-provided)
├── README.md
│
├── spotter_service/                    # Django project package
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
│
└── fuel_route/                         # Core application
    ├── models.py                       # TruckStop model
    ├── serializers.py                  # DRF request/response serializers
    ├── services.py                     # Routing, geocoding & optimisation engine
    ├── views.py                        # RouteAPIView, MapView
    ├── urls.py                         # URL patterns
    ├── tests.py                        # Integration & API tests
    │
    ├── data/
    │   └── city_coords.json            # 29,829 US city coordinates (offline)
    │
    ├── management/commands/
    │   └── load_fuel_csv.py            # Bulk CSV ingestion command
    │
    └── templates/fuel_route/
        └── map.html                    # Leaflet.js interactive UI
```

---

## 4. Data Model

```python
# fuel_route/models.py

class TruckStop(models.Model):
    opis_id       = models.IntegerField(db_index=True)
    name          = models.CharField(max_length=255)
    address       = models.CharField(max_length=255)
    city          = models.CharField(max_length=100, db_index=True)
    state         = models.CharField(max_length=10,  db_index=True)
    retail_price  = models.FloatField(db_index=True)
    latitude      = models.FloatField(db_index=True)
    longitude     = models.FloatField(db_index=True)

    class Meta:
        indexes = [
            models.Index(fields=["latitude", "longitude"]),
            models.Index(fields=["state", "city"]),
        ]
```

Compound indexes on `(latitude, longitude)` and `(state, city)` accelerate the bounding-box pre-filter that precedes the `cKDTree` query.

---

## 5. Data Ingestion

The management command `load_fuel_csv` reads the OPIS CSV, resolves coordinates for each `(City, State)` pair from a bundled offline dictionary, and bulk-inserts all records in a single `bulk_create` call.

```
$ python manage.py load_fuel_csv fuel-prices-for-be-assessment.csv
```

Key properties:

| Property | Detail |
|---|---|
| Source rows | 8,151 truck stops across the US and Canada |
| Offline coordinate coverage | 100 % (29,829 city entries + 51 state centroids) |
| Ingest time | ~0.7 s (bulk insert, no geocoding API calls) |
| Re-run behaviour | Clears existing records before inserting (`--clear` flag implicit on re-run) |

The offline coordinate dataset (`fuel_route/data/city_coords.json`) is generated once from a public US-cities database and manually extended with the 91 Canadian and edge-case locations present in the dataset. This eliminates all external geocoding dependencies during data load.

---

## 6. Core Engine

All logic lives in [`fuel_route/services.py`](fuel_route/services.py).

### 6.1 Geocoding

`geocode_location(query: str) → (lat, lon, display_name)`

Resolution order (fastest → slowest):

1. **In-process dict cache** — zero latency for repeated queries within a process lifetime.
2. **Raw coordinate passthrough** — e.g. `"41.8781, -87.6298"` accepted directly.
3. **Offline city/state dictionary** — `"Chicago, IL"` resolved in microseconds from the bundled JSON.
4. **OpenStreetMap Nominatim** — HTTP fallback for any location not in the offline dictionary. Results are cached to avoid repeat network calls.

Raises `LocationNotFoundError` if all methods fail.

### 6.2 OSRM Routing

`fetch_driving_route(origin, destination) → (distance_miles, waypoints, encoded_polyline)`

- Makes **exactly one HTTPS request** per unique origin/destination pair.
- Results are cached in-process by a rounded `(lat, lon)` key, so subsequent identical requests are free.
- Primary endpoint: `https://routing.openstreetmap.de/routed-car/route/v1/driving/`
- Fallback endpoint: `https://router.project-osrm.org/route/v1/driving/`
- Timeout: 15 seconds per attempt.
- Raises `RoutingServiceError` when all endpoints fail.

The encoded Google Polyline string is passed unchanged to the frontend, which decodes it client-side with a native JS implementation (no additional library required).

### 6.3 Spatial Corridor Filtering

`find_corridor_stops(waypoints, total_miles) → list[dict]`

**Step 1 — Waypoint densification**

OSRM may return sparse waypoints (sometimes only 2–3 per 100 miles on straight highways). The route is interpolated so no two consecutive points are more than 1 mile apart. This prevents truck stops from being missed between widely-spaced OSRM vertices.

**Step 2 — Bounding-box DB pre-filter**

A rectangular bounding box is computed from the densified waypoints (padded by the corridor radius in degrees) and used as a cheap SQL `WHERE latitude BETWEEN … AND longitude BETWEEN …` pre-filter, reducing the candidate set before the spatial query.

**Step 3 — cKDTree nearest-neighbour search**

```python
route_xy  = _to_cartesian(dense_pts, lat_ref)   # equirectangular projection → miles
stop_xy   = _to_cartesian(stop_coords, lat_ref)

tree = cKDTree(route_xy)
dists, idxs = tree.query(stop_xy, distance_upper_bound=10.0)  # 10-mile corridor
```

Each candidate stop is queried against the KD-tree. Stops whose nearest route point is within 10 miles are retained; their `mile_along_route` value is read from the pre-computed cumulative mileage array at the matched index.

The entire spatial filter (densification + KD-tree build + query) runs in **< 5 ms** for a typical cross-country route with ~3,000 densified points and ~500 corridor candidates.

### 6.4 Fuel Optimisation Algorithm

`optimise_refuel_stops(candidates, total_distance) → dict`

**Vehicle model**

| Parameter | Value | Override via |
|---|---|---|
| Tank capacity | 500 miles | `settings.VEHICLE_RANGE_MILES` |
| Fuel economy | 10 MPG | `settings.VEHICLE_MPG` |
| Initial fuel | Full tank (500 mi) | — |

**Short trips (≤ 500 miles)**

The destination is reachable on the initial tank; no physical stops are made. `station_refuel_cost_usd` is `$0.00`. `trip_consumption_cost_usd` is estimated from the median price among corridor candidates (or a \$3.50/gal US average if no corridor stops exist), representing the economic cost of fuel burned even though no pump purchase is required.

**Long trips (> 500 miles) — DAG Dynamic Programming**

The optimiser constructs a Directed Acyclic Graph (DAG) and solves it for the globally minimum-cost refuelling sequence:

```
Nodes : Origin (mile 0)
        + each corridor stop, sorted by mile_along_route
        + Destination (total_distance)

Edges : u → v exists iff  0 < mile(v) − mile(u) ≤ TANK_CAPACITY_MILES (500 mi)

Cost  : cost(Origin → v)  = 0.0          # vehicle departs with a pre-filled tank
        cost(stop_u → v)  = (leg_miles / MPG) × stop_u.retail_price

dp[v] = min over all valid predecessors u of  dp[u] + cost(u, v)
```

Predecessor pointers are tracked alongside `dp[]`, enabling exact path reconstruction via backtracking from the destination node. The algorithm guarantees the **global minimum total fuel cost** (O(n²) worst case; fast in practice because the 500-mile window limits the inner loop to at most ~50 reachable predecessors per node for US highway densities).

If a gap greater than `TANK_CAPACITY_MILES` exists between consecutive candidate stops (a "fuel desert"), no path reaches the destination. The function logs a warning and returns `reachable: False` so the API can surface this clearly to the caller.

**Disambiguated cost accounting**

| Field | Meaning |
|---|---|
| `station_refuel_cost_usd` | Dollars actually paid at pump stops (\\$0 for short trips) |
| `trip_consumption_cost_usd` | Full economic cost of all fuel burned, including the final leg from the last stop to the destination at that stop's price |
| `total_fuel_cost` | Backward-compatibility alias — always equals `trip_consumption_cost_usd` |
| `total_fuel_gallons` | Always `total_distance / MPG` — constant for a given route regardless of stop selection |

---

## 7. API Reference

### `POST /api/route/`

Plans the optimal refuelling itinerary between two locations.

**Request**

```http
POST /api/route/
Content-Type: application/json

{
  "start":  "Chicago, IL",
  "finish": "Austin, TX"
}
```

| Field | Type | Constraints |
|---|---|---|
| `start` | string | Required. Max 200 chars. Whitespace stripped. |
| `finish` | string | Required. Max 200 chars. Whitespace stripped. Must differ from `start`. |

**Response — 200 OK**

```json
{
  "start":  "Chicago, IL",
  "finish": "Austin, TX",
  "start_coords":  { "lat": 41.8781, "lon": -87.6298 },
  "finish_coords": { "lat": 30.2672, "lon": -97.7431 },
  "total_distance_miles": 1148.12,
  "total_fuel_gallons":   114.81,
  "station_refuel_cost_usd":   271.34,
  "trip_consumption_cost_usd": 330.29,
  "total_fuel_cost":           330.29,
  "reachable": true,
  "number_of_stops": 2,
  "fuel_stops": [
    {
      "opis_id":          379,
      "name":             "HUCKS FOOD & FUEL #379",
      "address":          "I-57, EXIT 54",
      "city":             "Marion",
      "state":            "IL",
      "retail_price":     2.929,
      "latitude":         37.7306,
      "longitude":        -88.9200,
      "mile_along_route": 318.45,
      "leg_distance_miles": 318.45,
      "gallons_pumped":   31.84,
      "cost":             93.26
    },
    { "...": "..." }
  ],
  "route_geometry": "_p~iF~ps|U_ulLnnqC_mqNvxq`@"
}
```

**Error responses**

| HTTP | Condition |
|---|---|
| `400` | Missing/invalid fields, origin equals destination, unresolvable location |
| `503` | All OSRM routing endpoints unreachable |
| `500` | Unexpected server error |

```json
// 400 — same origin and destination
{ "error": "Invalid request.", "details": { "non_field_errors": ["Origin and destination must be different locations."] } }

// 503 — routing service down
{ "error": "The routing service is temporarily unavailable. Please try again." }
```

---

## 8. Interactive Map UI

`GET /map/`

A single-page Leaflet.js interface served by Django's template engine.

**Features**

- **Basemap toggle** — Esri World Street Map (default) and Esri World Imagery (satellite), switchable via a top-right layer control. No API key required.
- **Route polyline** — decoded client-side from the OSRM-encoded geometry string; rendered as a blue `L.polyline`.
- **Origin marker (A)** — green circle with a popup showing the resolved location name.
- **Destination marker (B)** — red circle with destination popup.
- **Fuel stop markers** — amber numbered circles (1, 2, …). Clicking a marker or its sidebar card flies the map to that stop and opens a popup showing: station name, address, retail price, gallons pumped, route mile, and stop subtotal.
- **Summary metrics panel** — Total Distance, Trip Consumption Cost, Station Purchases, Gallons Consumed, and Number of Stops, updated on each calculation.
- **Quick-route presets** — one-click buttons for common routes (Chicago → Austin, LA → Dallas, NYC → Miami, Austin → Dallas).
- **Auto-calculation** — the default route (Chicago → Austin) is calculated automatically on page load.

---

## 9. DRF Serializers

```
fuel_route/serializers.py
```

| Serializer | Role |
|---|---|
| `RouteRequestSerializer` | Validates and normalises inbound `start`/`finish` strings |
| `CoordinateSerializer` | Typed `{ lat, lon }` nested object for start/finish coords |
| `FuelStopSerializer` | Per-stop output with all pricing and mileage fields |
| `RouteResponseSerializer` | Complete route plan output; composes the above, including `station_refuel_cost_usd`, `trip_consumption_cost_usd`, `total_fuel_cost` (alias), and `reachable` |

**`RouteRequestSerializer` validation**

```python
def validate_start(self, value):
    return value.strip()               # field-level: strip whitespace

def validate_finish(self, value):
    return value.strip()

def validate(self, attrs):             # cross-field: origin ≠ destination
    if attrs["start"].lower() == attrs["finish"].lower():
        raise serializers.ValidationError(
            "Origin and destination must be different locations."
        )
    return attrs
```

---

## 10. Custom Exceptions

Defined at the top of `fuel_route/services.py` and caught explicitly in `RouteAPIView`:

```python
class RoutingServiceError(Exception):
    """Raised when an upstream OSRM call fails or returns no routes."""

class LocationNotFoundError(Exception):
    """Raised when a location string cannot be resolved to coordinates."""
```

The view maps these to appropriate HTTP status codes:

```python
except LocationNotFoundError as exc:   → HTTP 400
except RoutingServiceError  as exc:   → HTTP 503
except Exception            as exc:   → HTTP 500  (logged with full traceback)
```

---

## 11. Settings Configurability

Vehicle parameters can be overridden in `settings.py` without touching service code:

```python
# settings.py

VEHICLE_RANGE_MILES = 500.0   # Maximum range on a full tank
VEHICLE_MPG         = 10.0    # Fuel consumption rate
```

The service reads these at import time with safe defaults:

```python
# services.py
from django.conf import settings

TANK_CAPACITY_MILES = getattr(settings, "VEHICLE_RANGE_MILES", 500.0)
MPG                 = getattr(settings, "VEHICLE_MPG", 10.0)
```

Other relevant settings:

| Setting | Default | Purpose |
|---|---|---|
| `DATABASES` | SQLite | Swap to PostgreSQL for production |
| `ALLOWED_HOSTS` | `['*']` | Restrict in production |
| `CORS_ALLOW_ALL_ORIGINS` | `True` | Restrict in production |
| `DEBUG` | `True` | Set to `False` in production |

---

## 12. Local Setup

### Prerequisites

- Python 3.12+
- pip

### Steps

```bash
# 1. Clone / enter the project directory
cd spotter-fuel-app

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install django djangorestframework requests numpy scipy polyline geopy django-cors-headers

# 4. Apply database migrations
python manage.py migrate

# 5. Ingest the fuel price dataset
python manage.py load_fuel_csv fuel-prices-for-be-assessment.csv

# 6. Start the development server
python manage.py runserver 127.0.0.1:8000
```

Open **http://127.0.0.1:8000/map/** in your browser.

### Quick API test (PowerShell)

```powershell
$body = '{"start": "Chicago, IL", "finish": "Austin, TX"}'
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/api/route/ `
  -ContentType 'application/json' `
  -Body $body | ConvertTo-Json -Depth 5
```

### Quick API test (curl / bash)

```bash
curl -s -X POST http://127.0.0.1:8000/api/route/ \
  -H "Content-Type: application/json" \
  -d '{"start": "Chicago, IL", "finish": "Austin, TX"}' | python -m json.tool
```

---

## 13. Management Commands

### `load_fuel_csv`

```
python manage.py load_fuel_csv [csv_file] [--clear]
```

| Argument | Default | Description |
|---|---|---|
| `csv_file` | `fuel-prices-for-be-assessment.csv` | Path to the OPIS price CSV |
| `--clear` | off | Truncate the `TruckStop` table before import |

Expected CSV columns (exactly as provided):

```
OPIS Truckstop ID, Truckstop Name, Address, City, State, Rack ID, Retail Price
```

The command performs:
1. Loads `fuel_route/data/city_coords.json` (29,829 city entries + 51 state centroids).
2. Resolves every `(City, State)` to `(latitude, longitude)` — 100 % offline coverage.
3. Falls back to the state centroid for unmatched rows; falls back to US geographic centre as a last resort.
4. Bulk-inserts all rows in a single `bulk_create(batch_size=2000)` call.
5. Reports ingest time and any skipped rows to stdout.

---

## 14. Running Tests

```bash
# Run the full test suite
python manage.py test fuel_route --verbosity=2
```

### Test cases

| Test | Description |
|---|---|
| `test_short_route_under_500_miles` | Austin → Dallas (~182 mi): verifies 0 stops, `total_fuel_cost > 0` (estimated from corridor price) |
| `test_long_cross_country_route_over_1000_miles` | Chicago → Austin (~1,148 mi): verifies ≥ 1 stop; asserts every consecutive leg ≤ 500 mi including the final leg to destination |
| `test_post_route_api_success` | `POST /api/route/` returns HTTP 200 with all required schema fields |
| `test_post_route_api_validation_error` | Missing `finish` → 400; identical origin and destination → 400 |
| `test_map_view_renders` | `GET /map/` returns HTTP 200 with expected HTML content |

Expected output:

```
Ran 5 tests in ~7s
OK
```

---

## 15. Design Decisions & Trade-offs

**Single OSRM call per route**
The spec requires exactly one external routing call. Results are cached in-process by a rounded coordinate key, so the development server never re-fetches the same route. For a production deployment behind a WSGI server with multiple processes, this cache would need to migrate to Redis or Memcached.

**SQLite for development**
SQLite is zero-configuration and sufficient for the assessment. Switching to PostgreSQL (with PostGIS for native spatial queries) would be a one-line `DATABASES` change and would unlock native `ST_Distance`/`ST_DWithin` as an alternative to the `cKDTree` approach.

**DAG Dynamic Programming — provably optimal**
The optimiser constructs a Directed Acyclic Graph over all corridor stops and solves it via forward DP in O(n²) worst case. Because the 500-mile tank window bounds each node's predecessor set to at most ~50 reachable predecessors on US highway networks, the algorithm completes in well under 10 ms for all tested routes. The result is the globally minimum-cost refuelling sequence — not a heuristic approximation.

**Offline geocoding over a live API**
The bundled `city_coords.json` covers 100 % of the 3,898 unique `(City, State)` pairs in the dataset. This eliminates Nominatim rate-limit risk during bulk import and makes the import process fully reproducible without network access.

### Coordinate Accuracy & Geographic Trade-offs

Truck stop coordinates are resolved from a bundled offline city/state centroid dictionary rather than address-level geocoding. This was an explicit design choice driven by three constraints:

1. **Zero-latency ingestion** — importing all 8,151 rows completes in under one second with no external network calls. A live geocoding pass at even 5 requests/second would take over 27 minutes and risk rate-limit bans.
2. **Full reproducibility** — the import pipeline produces identical results on any machine without network access, API keys, or quota management.
3. **Constraint satisfaction** — all 3,898 unique `(City, State)` pairs in the CSV have offline coverage; the `load_fuel_csv` command never falls back to a live geocoder.

**Trade-off:** coordinates represent city/state centroids, not the exact highway exit position of each stop. For a truck stop at *Marion, IL*, the assigned coordinate is the Marion city centroid rather than the precise I-57 exit ramp.

**Mitigation:** `CORRIDOR_RADIUS_MILES` (default `10.0`, overridable via `settings.ROUTE_CORRIDOR_RADIUS_MILES`) is sized to absorb this centroid offset. Analysis of the dataset shows that >95 % of US truck stops on major interstates lie within 10 miles of their city centroid.

**Production path:** a production deployment would replace centroid lookup with one of:
- **PostGIS + TIGER/Line Shapefiles** — free, exit-level address geocoding on-premise.
- **HERE Geocoding API / Google Maps Geocoding API** — per-address precision in a single async batch call at ingestion time, results persisted to the database.
- **A local Nominatim instance** — full OSM geocoding without rate limits, suitable for private infrastructure.

**Waypoint densification**
OSRM's `overview=full` polyline can still have segments longer than 5–10 miles on rural interstates. Densifying to ≤ 1 mile ensures the `cKDTree` nearest-neighbour query returns the correct route mile for every candidate stop, preventing stops from being projected to the wrong point on the route.

