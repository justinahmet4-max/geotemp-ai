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


@app.post("/api/analyze-temperature")
async def analyze_temperature(bounds: BoundingBox):
    return generate_mock_temperature(bounds)


@app.get("/")
async def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
