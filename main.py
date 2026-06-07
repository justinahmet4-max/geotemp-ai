from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import os

app = FastAPI(title="GeoTemp-AI", version="1.0.0")


class BoundingBox(BaseModel):
    north: float
    south: float
    east: float
    west: float


MONTHLY_BASE_TEMPS = {
    "Ocak": -2, "Şubat": 1, "Mart": 6, "Nisan": 11,
    "Mayıs": 16, "Haziran": 21, "Temmuz": 24,
    "Ağustos": 24, "Eylül": 20, "Ekim": 14,
    "Kasım": 7, "Aralık": 1
}


def generate_mock_temperature(bounds: BoundingBox) -> dict:
    np.random.seed(
        int((bounds.north + bounds.south + bounds.east + bounds.west) * 1000) % 2**31
    )

    lat_factor = (bounds.north + bounds.south) / 2
    lon_factor = (bounds.east + bounds.west) / 2
    base_offset = (lat_factor - 39) * -0.8 + (lon_factor - 32) * 0.3

    monthly_data = []
    for month, base_temp in MONTHLY_BASE_TEMPS.items():
        noise = np.random.uniform(-2.5, 2.5)
        temp = round(base_temp + base_offset + noise, 1)
        monthly_data.append({"ay": month, "sicaklik": temp})

    region_names = [
        "Anadolu Bölgesi", "İç Anadolu", "Batı Karadeniz",
        "Ege Kıyıları", "Akdeniz Havzası", "Doğu Anadolu",
        "Marmara Bölgesi", "Güneydoğu Anadolu"
    ]
    region_idx = int(abs(lat_factor * lon_factor)) % len(region_names)

    return {
        "region": region_names[region_idx],
        "bounds": {
            "north": bounds.north,
            "south": bounds.south,
            "east": bounds.east,
            "west": bounds.west
        },
        "monthly_data": monthly_data
    }


def generate_thermal_grid(bounds: BoundingBox) -> dict:
    import random
    grid_size = 30
    seed_val = int((bounds.north + bounds.south + bounds.east + bounds.west) * 1000) % (2**31)
    random.seed(seed_val)

    lat_factor = (bounds.north + bounds.south) / 2
    lon_factor = (bounds.east + bounds.west) / 2
    base_offset = (lat_factor - 39) * -0.8 + (lon_factor - 32) * 0.3

    hotspots = []
    for _ in range(random.randint(2, 5)):
        hotspots.append({
            "lat": bounds.south + random.random() * (bounds.north - bounds.south),
            "lng": bounds.west + random.random() * (bounds.east - bounds.west),
            "intensity": random.uniform(-6, 8),
            "radius": random.uniform(0.15, 0.45)
        })

    grid = []
    all_temps = []
    for i in range(grid_size):
        for j in range(grid_size):
            lat = bounds.south + (bounds.north - bounds.south) * i / (grid_size - 1)
            lng = bounds.west + (bounds.east - bounds.west) * j / (grid_size - 1)

            base = 20 + base_offset
            grad_lat = -2 + 4 * i / (grid_size - 1)
            grad_lon = -1.5 + 3 * j / (grid_size - 1)

            hotspot_effect = 0
            for hs in hotspots:
                dist = ((lat - hs["lat"]) ** 2 + (lng - hs["lng"]) ** 2) ** 0.5
                hotspot_effect += hs["intensity"] * max(0, 1 - dist / hs["radius"])

            noise = random.gauss(0, 1.2)
            temp = round(base + grad_lat + grad_lon + hotspot_effect + noise, 1)
            all_temps.append(temp)
            grid.append(temp)

    return {
        "bounds": {
            "north": float(bounds.north),
            "south": float(bounds.south),
            "east": float(bounds.east),
            "west": float(bounds.west)
        },
        "grid_size": grid_size,
        "grid": grid,
        "min_temp": round(min(all_temps), 1),
        "max_temp": round(max(all_temps), 1)
    }


@app.post("/api/analyze-temperature")
async def analyze_temperature(bounds: BoundingBox):
    return generate_mock_temperature(bounds)


@app.post("/api/thermal-grid")
async def thermal_grid(bounds: BoundingBox):
    return generate_thermal_grid(bounds)


@app.get("/")
async def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
