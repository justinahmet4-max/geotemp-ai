from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import os
import json
import asyncio
import random
import math

app = FastAPI(title="GeoTemp-AI", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class BoundingBox(BaseModel):
    north: float
    south: float
    east: float
    west: float


class PointRequest(BaseModel):
    lat: float
    lng: float


MONTHLY_BASE_TEMPS = {
    "Ocak": -2, "Şubat": 1, "Mart": 6, "Nisan": 11,
    "Mayıs": 16, "Haziran": 21, "Temmuz": 24,
    "Ağustos": 24, "Eylül": 20, "Ekim": 14,
    "Kasım": 7, "Aralık": 1
}

ZEMIN_SINIFLARI = {
    "ZA": {"ad": "Zemin Sinifi ZA", "aciklama": "Kayali zemin, yuzeye yakin", "vs30_min": 800, "vs30_max": 1500, "renk": "#2d5a27"},
    "ZB": {"ad": "Zemin Sinifi ZB", "aciklama": "Sert killi/kumlu tabaka", "vs30_min": 500, "vs30_max": 800, "renk": "#6b8e5a"},
    "ZC": {"ad": "Zemin Sinifi ZC", "aciklama": "Orta sertlikte killi zemin", "vs30_min": 300, "vs30_max": 500, "renk": "#c4a882"},
    "ZD": {"ad": "Zemin Sinifi ZD", "aciklama": "Yumuşak killi/alüvyon zemin", "vs30_min": 150, "vs30_max": 300, "renk": "#d4a574"},
    "ZE": {"ad": "Zemin Sinifi ZE", "aciklama": "Cok yumusak balcikli zemin", "vs30_min": 0, "vs30_max": 150, "renk": "#cc6633"},
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
        "bounds": {"north": bounds.north, "south": bounds.south, "east": bounds.east, "west": bounds.west},
        "monthly_data": monthly_data
    }


def generate_thermal_grid(bounds: BoundingBox) -> dict:
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
        "bounds": {"north": float(bounds.north), "south": float(bounds.south), "east": float(bounds.east), "west": float(bounds.west)},
        "grid_size": grid_size, "grid": grid,
        "min_temp": round(min(all_temps), 1), "max_temp": round(max(all_temps), 1)
    }


def generate_geological_data(bounds: BoundingBox) -> dict:
    seed_val = int((bounds.north + bounds.south + bounds.east + bounds.west) * 1000) % (2**31)
    random.seed(seed_val)
    lat_factor = (bounds.north + bounds.south) / 2
    lon_factor = (bounds.east + bounds.west) / 2

    sinif_keys = list(ZEMIN_SINIFLARI.keys())
    sinif_idx = int(abs(math.sin(lat_factor * 0.1) * math.cos(lon_factor * 0.08)) * 100) % len(sinif_keys)
    ana_sinif = sinif_keys[sinif_idx]

    vs30 = round(random.uniform(ZEMIN_SINIFLARI[ana_sinif]["vs30_min"], ZEMIN_SINIFLARI[ana_sinif]["vs30_max"]), 0)

    heyelan_riski = round(min(1.0, max(0.0, abs(math.sin(lat_factor * 0.05)) * 0.6 + random.uniform(-0.2, 0.3))), 2)
    aluvyon_kalinlik = round(random.uniform(2, 50), 1)
    yeralti_su_derinligi = round(random.uniform(5, 120), 1)

    grid_size = 20
    zemin_grid = []
    for i in range(grid_size):
        for j in range(grid_size):
            lat = bounds.south + (bounds.north - bounds.south) * i / (grid_size - 1)
            lng = bounds.west + (bounds.east - bounds.west) * j / (grid_size - 1)
            idx = int(abs(math.sin(lat * 0.15) * math.cos(lng * 0.12)) * 100) % len(sinif_keys)
            sinif = sinif_keys[idx]
            vs30_val = round(random.uniform(ZEMIN_SINIFLARI[sinif]["vs30_min"], ZEMIN_SINIFLARI[sinif]["vs30_max"]), 0)
            zemin_grid.append({
                "sinif": sinif,
                "vs30": vs30_val,
                "renk": ZEMIN_SINIFLARI[sinif]["renk"]
            })

    return {
        "bounds": {"north": bounds.north, "south": bounds.south, "east": bounds.east, "west": bounds.west},
        "ana_sinif": ana_sinif,
        "sinif_bilgisi": ZEMIN_SINIFLARI[ana_sinif],
        "vs30": vs30,
        "heyelan_riski": heyelan_riski,
        "aluvyon_kalinlik": aluvyon_kalinlik,
        "yeralti_su_derinligi": yeralti_su_derinligi,
        "grid_size": grid_size,
        "zemin_grid": zemin_grid
    }


def generate_spectral_data(bounds: BoundingBox) -> dict:
    seed_val = int((bounds.north + bounds.south + bounds.east + bounds.west) * 1000) % (2**31)
    random.seed(seed_val)
    grid_size = 30
    ndvi_grid = []
    ndwi_grid = []
    mineral_grid = []
    surface_temp = []

    for i in range(grid_size):
        for j in range(grid_size):
            lat = bounds.south + (bounds.north - bounds.south) * i / (grid_size - 1)
            lng = bounds.west + (bounds.east - bounds.west) * j / (grid_size - 1)

            ndvi = max(0, min(1, 0.3 + 0.4 * math.sin(lat * 0.1) * math.cos(lng * 0.15) + random.gauss(0, 0.1)))
            ndwi = max(0, min(1, 0.2 + 0.3 * math.cos(lat * 0.08) * math.sin(lng * 0.12) + random.gauss(0, 0.08)))
            mineral = max(0, min(1, 0.5 + 0.3 * math.sin(lat * 0.2 + lng * 0.1) + random.gauss(0, 0.1)))
            s_temp = max(0, min(1, 0.4 + 0.3 * math.cos(lat * 0.06) + random.gauss(0, 0.1)))

            ndvi_grid.append(round(ndvi, 3))
            ndwi_grid.append(round(ndwi, 3))
            mineral_grid.append(round(mineral, 3))
            surface_temp.append(round(s_temp, 3))

    return {
        "bounds": {"north": bounds.north, "south": bounds.south, "east": bounds.east, "west": bounds.west},
        "grid_size": grid_size,
        "ndvi": {"grid": ndvi_grid, "min": round(min(ndvi_grid), 3), "max": round(max(ndvi_grid), 3)},
        "ndwi": {"grid": ndwi_grid, "min": round(min(ndwi_grid), 3), "max": round(max(ndwi_grid), 3)},
        "mineral": {"grid": mineral_grid, "min": round(min(mineral_grid), 3), "max": round(max(mineral_grid), 3)},
        "surface_temp": {"grid": surface_temp, "min": round(min(surface_temp), 3), "max": round(max(surface_temp), 3)},
        "yorum": {
            "bitki_durumu": "Iyi" if ndvi_grid[len(ndvi_grid)//2] > 0.5 else "Orta" if ndvi_grid[len(ndvi_grid)//2] > 0.25 else "Dusuk",
            "nem_durumu": "Yuksek" if ndwi_grid[len(ndwi_grid)//2] > 0.5 else "Orta" if ndwi_grid[len(ndwi_grid)//2] > 0.25 else "Dusuk",
            "mineral_bilesim": "Zengin" if mineral_grid[len(mineral_grid)//2] > 0.6 else "Orta" if mineral_grid[len(mineral_grid)//2] > 0.3 else "Fakir"
        }
    }


def generate_gpr_profile(lat: float, lng: float) -> dict:
    seed_val = int((lat + lng) * 10000) % (2**31)
    random.seed(seed_val)
    depth_points = 100
    time_window = 100

    signal = []
    depth = []
    for t in range(time_window):
        d = t * 0.5
        depth.append(round(d, 1))
        amp = 0
        for layer in range(5):
            layer_depth = random.uniform(10, 80) * (layer + 1)
            layer_amp = random.uniform(0.3, 0.9) * math.exp(-d / 200) * math.sin(2 * math.pi * d / (layer_depth + 10))
            amp += layer_amp
        amp += random.gauss(0, 0.05)
        amp = max(-1, min(1, amp))
        signal.append(round(amp, 4))

    katmanlar = [
        {"ad": "Toprak Yuzey", "derinlik": round(random.uniform(0, 10), 1), "kalınlık": round(random.uniform(5, 20), 1)},
        {"ad": "Killi Tabaka", "derinlik": round(random.uniform(15, 30), 1), "kalınlık": round(random.uniform(10, 30), 1)},
        {"ad": "Kumlu Tabaka", "derinlik": round(random.uniform(40, 60), 1), "kalınlık": round(random.uniform(15, 40), 1)},
        {"ad": "Kaya Ana Kaya", "derinlik": round(random.uniform(80, 150), 1), "kalınlık": 0},
        {"ad": "Yeralti Suyu", "derinlik": round(random.uniform(30, 100), 1), "kalınlık": round(random.uniform(5, 25), 1)},
    ]

    return {
        "lat": lat, "lng": lng,
        "time_window_us": time_window,
        "depth": depth,
        "signal": signal,
        "katmanlar": katmanlar,
        "ornek_sayisi": depth_points
    }


@app.post("/api/analyze-temperature")
async def analyze_temperature(bounds: BoundingBox):
    return generate_mock_temperature(bounds)


@app.post("/api/thermal-grid")
async def thermal_grid(bounds: BoundingBox):
    return generate_thermal_grid(bounds)


@app.post("/api/geological-data")
async def geological_data(bounds: BoundingBox):
    return generate_geological_data(bounds)


@app.post("/api/spectral-analysis")
async def spectral_analysis(bounds: BoundingBox):
    return generate_spectral_data(bounds)


@app.post("/api/gpr-scan")
async def gpr_scan(point: PointRequest):
    return generate_gpr_profile(point.lat, point.lng)


@app.websocket("/ws/gpr")
async def websocket_gpr(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            params = json.loads(data)
            lat = params.get("lat", 39.9)
            lng = params.get("lng", 32.8)
            depth_points = 100

            for t in range(depth_points):
                amp = 0
                for layer in range(5):
                    layer_depth = random.uniform(10, 80) * (layer + 1)
                    amp += random.uniform(0.3, 0.9) * math.exp(-t * 0.5 / 200) * math.sin(2 * math.pi * t * 0.5 / (layer_depth + 10))
                amp += random.gauss(0, 0.05)
                amp = max(-1, min(1, amp))
                await websocket.send_json({
                    "type": "signal",
                    "time": t,
                    "depth": round(t * 0.5, 1),
                    "amplitude": round(amp, 4),
                    "progress": round((t + 1) / depth_points * 100, 1)
                })
                await asyncio.sleep(0.02)

            await websocket.send_json({"type": "complete", "message": "Tarama tamamlandi"})
    except WebSocketDisconnect:
        pass


@app.get("/")
async def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
