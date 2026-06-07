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
    grid_size = 12
    seed_val = int((bounds.north + bounds.south + bounds.east + bounds.west) * 1000) % 2**31
    np.random.seed(seed_val)

    lat_factor = (bounds.north + bounds.south) / 2
    lon_factor = (bounds.east + bounds.west) / 2
    base_offset = (lat_factor - 39) * -0.8 + (lon_factor - 32) * 0.3

    lats = np.linspace(bounds.south, bounds.north, grid_size)
    lngs = np.linspace(bounds.west, bounds.east, grid_size)

    noise_field = np.random.uniform(-3, 3, (grid_size, grid_size))
    gradient_lat = np.linspace(-2, 2, grid_size).reshape(-1, 1)
    gradient_lon = np.linspace(-1.5, 1.5, grid_size).reshape(1, -1)

    grid = []
    for i in range(grid_size):
        for j in range(grid_size):
            temp = round(
                20 + base_offset
                + float(gradient_lat[i])
                + float(gradient_lon[j])
                + float(noise_field[i, j]),
                1
            )
            grid.append({
                "lat": round(float(lats[i]), 6),
                "lng": round(float(lngs[j]), 6),
                "sicaklik": temp
            })

    return {
        "bounds": {
            "north": bounds.north,
            "south": bounds.south,
            "east": bounds.east,
            "west": bounds.west
        },
        "grid_size": grid_size,
        "grid": grid,
        "min_temp": round(float(np.min([g["sicaklik"] for g in grid])), 1),
        "max_temp": round(float(np.max([g["sicaklik"] for g in grid])), 1)
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
