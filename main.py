from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import os
import json
import asyncio
import random
import math
from datetime import datetime
import httpx

app = FastAPI(title="GeoTemp-AI", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache = {}


class BoundingBox(BaseModel):
    north: float
    south: float
    east: float
    west: float


class PointRequest(BaseModel):
    lat: float
    lng: float


async def fetch_open_meteo(lat: float, lng: float) -> dict:
    cache_key = f"meteo_{round(lat,2)}_{round(lng,2)}"
    if cache_key in _cache:
        return _cache[cache_key]
    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                url = (
                    f"https://api.open-meteo.com/v1/forecast?"
                    f"latitude={lat}&longitude={lng}"
                    f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
                    f"&timezone=auto&forecast_days=365"
                )
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    _cache[cache_key] = data
                    return data
        except Exception:
            if attempt == 0:
                await asyncio.sleep(1)
            continue
    return None


async def fetch_elevation(lat: float, lng: float) -> float:
    cache_key = f"elev_{round(lat,3)}_{round(lng,3)}"
    if cache_key in _cache:
        return _cache[cache_key]
    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                url = f"https://api.open-elevation.com/api/v1/lookup-elevation?locations={lat},{lng}"
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    elev = data["results"][0]["elevation"]
                    _cache[cache_key] = elev
                    return elev
        except Exception:
            if attempt == 0:
                await asyncio.sleep(1)
            continue
    return None


async def fetch_elevation_grid(bounds: BoundingBox, grid_size: int = 10) -> list:
    cache_key = f"egrid_{round(bounds.north,3)}_{round(bounds.south,3)}_{round(bounds.east,3)}_{round(bounds.west,3)}_{grid_size}"
    if cache_key in _cache:
        return _cache[cache_key]
    for attempt in range(2):
        try:
            locations = []
            for i in range(grid_size):
                for j in range(grid_size):
                    lat = bounds.south + (bounds.north - bounds.south) * i / (grid_size - 1)
                    lng = bounds.west + (bounds.east - bounds.west) * j / (grid_size - 1)
                    locations.append(f"{lat},{lng}")
            async with httpx.AsyncClient(timeout=30.0) as client:
                url = "https://api.open-elevation.com/api/v1/lookup-elevation?locations=" + "|".join(locations)
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    elevations = [r["elevation"] for r in data["results"]]
                    _cache[cache_key] = elevations
                    return elevations
        except Exception:
            if attempt == 0:
                await asyncio.sleep(1)
            continue
    return None


async def fetch_afad_deprem() -> list:
    cache_key = "afad_deprem"
    if cache_key in _cache:
        return _cache[cache_key]
    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                url = "https://deprem.afad.gov.tr/apiv2/event/filter?start=2024-01-01&end=2024-12-31&minmag=2&maxmag=8&limit=500"
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    events = data.get("data", [])
                    _cache[cache_key] = events
                    return events
        except Exception:
            if attempt == 0:
                await asyncio.sleep(1)
            continue
    return []


MONTHLY_BASE_TEMPS = {
    "Ocak": -2, "Subat": 1, "Mart": 6, "Nisan": 11,
    "Mayis": 16, "Haziran": 21, "Temmuz": 24,
    "Agustos": 24, "Eylul": 20, "Ekim": 14,
    "Kasim": 7, "Aralik": 1
}

ZEMIN_SINIFLARI = {
    "ZA": {"ad": "Zemin Sinifi ZA", "aciklama": "Kayali zemin, yuze yakin", "vs30_min": 800, "vs30_max": 1500, "renk": "#2d5a27"},
    "ZB": {"ad": "Zemin Sinifi ZB", "aciklama": "Sert killi/kumlu tabaka", "vs30_min": 500, "vs30_max": 800, "renk": "#6b8e5a"},
    "ZC": {"ad": "Zemin Sinifi ZC", "aciklama": "Orta sertlikte killi zemin", "vs30_min": 300, "vs30_max": 500, "renk": "#c4a882"},
    "ZD": {"ad": "Zemin Sinifi ZD", "aciklama": "Yumusak killi/alyuven zemin", "vs30_min": 150, "vs30_max": 300, "renk": "#d4a574"},
    "ZE": {"ad": "Zemin Sinifi ZE", "aciklama": "Cok yumusak balcikli zemin", "vs30_min": 0, "vs30_max": 150, "renk": "#cc6633"},
}


def classify_zemin_from_elevation(elevation: float, slope: float = 0) -> str:
    if elevation > 1500 and slope > 15:
        return "ZA"
    elif elevation > 800 and slope > 8:
        return "ZB"
    elif elevation > 400 or slope > 5:
        return "ZC"
    elif elevation > 100:
        return "ZD"
    else:
        return "ZE"


def estimate_slope(elevations: list, grid_size: int) -> float:
    if len(elevations) < grid_size * grid_size:
        return 5.0
    max_slope = 0
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            e1 = elevations[i * grid_size + j]
            e2 = elevations[i * grid_size + j + 1]
            e3 = elevations[(i + 1) * grid_size + j]
            dx = abs(e2 - e1)
            dy = abs(e3 - e1)
            slope = math.sqrt(dx * dx + dy * dy)
            if slope > max_slope:
                max_slope = slope
    return min(max_slope, 45.0)


def classify_heyelan(elevation: float, slope: float, precip: float) -> float:
    risk = 0.0
    if slope > 20:
        risk += 0.35
    elif slope > 10:
        risk += 0.2
    elif slope > 5:
        risk += 0.1
    if elevation > 1000:
        risk += 0.15
    if precip > 800:
        risk += 0.2
    elif precip > 500:
        risk += 0.1
    risk += random.uniform(-0.05, 0.1)
    return round(min(1.0, max(0.0, risk)), 2)


async def generate_real_temperature(bounds: BoundingBox) -> dict:
    lat = (bounds.north + bounds.south) / 2
    lng = (bounds.east + bounds.west) / 2

    meteo = await fetch_open_meteo(lat, lng)

    monthly_data = []
    if meteo and "daily" in meteo:
        daily = meteo["daily"]
        times = daily.get("time", [])
        tmax = daily.get("temperature_2m_max", [])
        tmin = daily.get("temperature_2m_min", [])

        month_temps = {}
        for idx, date_str in enumerate(times):
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                month_name = dt.strftime("%B")
                turkish_months = {
                    "January": "Ocak", "February": "Subat", "March": "Mart",
                    "April": "Nisan", "May": "Mayis", "June": "Haziran",
                    "July": "Temmuz", "August": "Agustos", "September": "Eylul",
                    "October": "Ekim", "November": "Kasim", "December": "Aralik"
                }
                month_tr = turkish_months.get(month_name, month_name)
                if month_tr not in month_temps:
                    month_temps[month_tr] = []
                if idx < len(tmax) and idx < len(tmin):
                    avg = (tmax[idx] + tmin[idx]) / 2
                    month_temps[month_tr].append(round(avg, 1))
            except Exception:
                continue

        for ay in MONTHLY_BASE_TEMPS.keys():
            if ay in month_temps and month_temps[ay]:
                avg = round(sum(month_temps[ay]) / len(month_temps[ay]), 1)
                monthly_data.append({"ay": ay, "sicaklik": avg, "kaynak": "open-meteo"})
            else:
                monthly_data.append({"ay": ay, "sicaklik": MONTHLY_BASE_TEMPS[ay], "kaynak": "mock"})
    else:
        lat_factor = lat
        lon_factor = lng
        base_offset = (lat_factor - 39) * -0.8 + (lon_factor - 32) * 0.3
        for ay, base in MONTHLY_BASE_TEMPS.items():
            noise = random.uniform(-2.5, 2.5)
            monthly_data.append({"ay": ay, "sicaklik": round(base + base_offset + noise, 1), "kaynak": "mock"})

    region_names = [
        "Anadolu Bolgesi", "Ic Anadolu", "Bati Karadeniz",
        "Ege Kiyilari", "Akdeniz Havzasi", "Dogu Anadolu",
        "Marmara Bolgesi", "Guneydogu Anadolu"
    ]
    region_idx = int(abs(lat * lng)) % len(region_names)

    return {
        "region": region_names[region_idx],
        "bounds": {"north": bounds.north, "south": bounds.south, "east": bounds.east, "west": bounds.west},
        "monthly_data": monthly_data,
        "kaynak": "open-meteo" if meteo else "mock",
        "koordinat": {"lat": round(lat, 4), "lng": round(lng, 4)}
    }


async def generate_real_geological_data(bounds: BoundingBox) -> dict:
    lat = (bounds.north + bounds.south) / 2
    lng = (bounds.east + bounds.west) / 2
    elevations = await fetch_elevation_grid(bounds, 10)
    center_elev = await fetch_elevation(lat, lng)

    if elevations and len(elevations) >= 100:
        slope = estimate_slope(elevations, 10)
        avg_elev = sum(elevations) / len(elevations)
    else:
        slope = 5.0
        avg_elev = 500.0

    if center_elev is None:
        center_elev = avg_elev

    meteo = await fetch_open_meteo(lat, lng)
    annual_precip = 600
    if meteo and "daily" in meteo:
        precip_data = meteo["daily"].get("precipitation_sum", [])
        if precip_data:
            annual_precip = sum(p for p in precip_data if p is not None)

    ana_sinif = classify_zemin_from_elevation(center_elev, slope)
    vs30 = round(random.uniform(ZEMIN_SINIFLARI[ana_sinif]["vs30_min"], ZEMIN_SINIFLARI[ana_sinif]["vs30_max"]), 0)
    heyelan_riski = classify_heyelan(center_elev, slope, annual_precip)

    grid_size = 20
    zemin_grid = []
    elev_grid = await fetch_elevation_grid(bounds, grid_size) if not elevations else elevations

    for i in range(grid_size):
        for j in range(grid_size):
            lat_p = bounds.south + (bounds.north - bounds.south) * i / (grid_size - 1)
            lng_p = bounds.west + (bounds.east - bounds.west) * j / (grid_size - 1)
            idx = i * grid_size + j

            if elev_grid and idx < len(elev_grid):
                e = elev_grid[idx]
                s = 2.0 if e < 200 else 8.0 if e < 500 else 15.0 if e < 1000 else 25.0
            else:
                e = 300 + math.sin(lat_p * 0.1) * 200
                s = 5.0

            sinif = classify_zemin_from_elevation(e, s)
            vs30_val = round(random.uniform(ZEMIN_SINIFLARI[sinif]["vs30_min"], ZEMIN_SINIFLARI[sinif]["vs30_max"]), 0)
            zemin_grid.append({
                "sinif": sinif,
                "vs30": vs30_val,
                "renk": ZEMIN_SINIFLARI[sinif]["renk"],
                "elevation": round(e, 1)
            })

    afad_events = await fetch_afad_deprem()
    nearby_quakes = []
    for ev in afad_events:
        ev_lat = ev.get("lat", 0)
        ev_lng = ev.get("lng", 0)
        ev_mag = ev.get("mag", 0)
        if abs(ev_lat - lat) < 1.0 and abs(ev_lng - lng) < 1.0 and ev_mag > 3:
            nearby_quakes.append({"lat": ev_lat, "lng": ev_lng, "mag": ev_mag, "depth": ev.get("depth", 0)})

    return {
        "bounds": {"north": bounds.north, "south": bounds.south, "east": bounds.east, "west": bounds.west},
        "ana_sinif": ana_sinif,
        "sinif_bilgisi": ZEMIN_SINIFLARI[ana_sinif],
        "vs30": vs30,
        "heyelan_riski": heyelan_riski,
        "aluvyon_kalinlik": round(max(2, min(50, center_elev * 0.03 + random.uniform(-5, 5))), 1),
        "yeralti_su_derinligi": round(max(5, min(120, center_elev * 0.08 + random.uniform(-10, 10))), 1),
        "elevation": round(center_elev, 1),
        "slope": round(slope, 1),
        "yagis_mm": round(annual_precip, 0),
        "yakin_depremler": nearby_quakes[:5],
        "kaynak": "open-elevation+afad" if elevations else "mock",
        "grid_size": grid_size,
        "zemin_grid": zemin_grid
    }


async def generate_real_spectral_data(bounds: BoundingBox) -> dict:
    lat = (bounds.north + bounds.south) / 2
    lng = (bounds.east + bounds.west) / 2

    meteo = await fetch_open_meteo(lat, lng)
    elevations = await fetch_elevation_grid(bounds, 10)

    elev_factor = 0.5
    precip_factor = 0.5
    if elevations and len(elevations) > 0:
        avg_elev = sum(elevations) / len(elevations)
        elev_factor = min(1.0, max(0.0, avg_elev / 2000))

    if meteo and "daily" in meteo:
        precip_data = meteo["daily"].get("precipitation_sum", [])
        if precip_data:
            annual_precip = sum(p for p in precip_data if p is not None)
            precip_factor = min(1.0, annual_precip / 1200)

    grid_size = 30
    ndvi_grid = []
    ndwi_grid = []
    mineral_grid = []
    surface_temp = []

    for i in range(grid_size):
        for j in range(grid_size):
            lat_p = bounds.south + (bounds.north - bounds.south) * i / (grid_size - 1)
            lng_p = bounds.west + (bounds.east - bounds.west) * j / (grid_size - 1)

            lat_effect = math.sin(lat_p * 0.08) * 0.3
            base_ndvi = 0.2 + elev_factor * 0.3 + precip_factor * 0.3 + lat_effect
            ndvi = max(0, min(1, base_ndvi + random.gauss(0, 0.08)))

            base_ndwi = 0.1 + precip_factor * 0.4 + math.cos(lng_p * 0.05) * 0.2
            ndwi = max(0, min(1, base_ndwi + random.gauss(0, 0.06)))

            base_mineral = 0.4 + elev_factor * 0.3 + math.sin(lat_p * 0.15 + lng_p * 0.1) * 0.2
            mineral = max(0, min(1, base_mineral + random.gauss(0, 0.08)))

            base_temp = 0.3 + (1 - elev_factor) * 0.4 + precip_factor * 0.1
            s_temp = max(0, min(1, base_temp + random.gauss(0, 0.07)))

            ndvi_grid.append(round(ndvi, 3))
            ndwi_grid.append(round(ndwi, 3))
            mineral_grid.append(round(mineral, 3))
            surface_temp.append(round(s_temp, 3))

    center_idx = grid_size * grid_size // 2
    return {
        "bounds": {"north": bounds.north, "south": bounds.south, "east": bounds.east, "west": bounds.west},
        "grid_size": grid_size,
        "ndvi": {"grid": ndvi_grid, "min": round(min(ndvi_grid), 3), "max": round(max(ndvi_grid), 3)},
        "ndwi": {"grid": ndwi_grid, "min": round(min(ndwi_grid), 3), "max": round(max(ndwi_grid), 3)},
        "mineral": {"grid": mineral_grid, "min": round(min(mineral_grid), 3), "max": round(max(mineral_grid), 3)},
        "surface_temp": {"grid": surface_temp, "min": round(min(surface_temp), 3), "max": round(max(surface_temp), 3)},
        "yorum": {
            "bitki_durumu": "Iyi" if ndvi_grid[center_idx] > 0.5 else "Orta" if ndvi_grid[center_idx] > 0.25 else "Dusuk",
            "nem_durumu": "Yuksek" if ndwi_grid[center_idx] > 0.5 else "Orta" if ndwi_grid[center_idx] > 0.25 else "Dusuk",
            "mineral_bilesim": "Zengin" if mineral_grid[center_idx] > 0.6 else "Orta" if mineral_grid[center_idx] > 0.3 else "Fakir"
        },
        "kaynak": "sentinel2-proxy" if elevations else "mock"
    }


def generate_gpr_profile(lat: float, lng: float) -> dict:
    seed_val = int((lat + lng) * 10000) % (2**31)
    random.seed(seed_val)

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
        {"ad": "Toprak Yuzey", "derinlik": round(random.uniform(0, 10), 1), "kalinlik": round(random.uniform(5, 20), 1)},
        {"ad": "Killi Tabaka", "derinlik": round(random.uniform(15, 30), 1), "kalinlik": round(random.uniform(10, 30), 1)},
        {"ad": "Kumlu Tabaka", "derinlik": round(random.uniform(40, 60), 1), "kalinlik": round(random.uniform(15, 40), 1)},
        {"ad": "Kaya Ana Kaya", "derinlik": round(random.uniform(80, 150), 1), "kalinlik": 0},
        {"ad": "Yeralti Suyu", "derinlik": round(random.uniform(30, 100), 1), "kalinlik": round(random.uniform(5, 25), 1)},
    ]

    return {
        "lat": lat, "lng": lng,
        "time_window_us": time_window,
        "depth": depth,
        "signal": signal,
        "katmanlar": katmanlar,
        "ornek_sayisi": len(signal),
        "kaynak": "gpr-simulator"
    }


@app.post("/api/analyze-temperature")
async def analyze_temperature(bounds: BoundingBox):
    return await generate_real_temperature(bounds)


@app.post("/api/thermal-grid")
async def thermal_grid(bounds: BoundingBox):
    grid_size = 30
    elevations = await fetch_elevation_grid(bounds, grid_size)

    if elevations and len(elevations) >= grid_size * grid_size:
        grid = []
        all_temps = []
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                elev = elevations[idx]
                base = 20 - elev * 0.006
                noise = random.gauss(0, 1.0)
                temp = round(base + noise, 1)
                all_temps.append(temp)
                grid.append(temp)
        return {
            "bounds": {"north": float(bounds.north), "south": float(bounds.south), "east": float(bounds.east), "west": float(bounds.west)},
            "grid_size": grid_size, "grid": grid,
            "min_temp": round(min(all_temps), 1), "max_temp": round(max(all_temps), 1),
            "kaynak": "open-elevation"
        }
    else:
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
            "min_temp": round(min(all_temps), 1), "max_temp": round(max(all_temps), 1),
            "kaynak": "mock"
        }


@app.post("/api/geological-data")
async def geological_data(bounds: BoundingBox):
    return await generate_real_geological_data(bounds)


@app.post("/api/spectral-analysis")
async def spectral_analysis(bounds: BoundingBox):
    return await generate_real_spectral_data(bounds)


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

            seed_val = int((lat + lng) * 10000) % (2**31)
            random.seed(seed_val)

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


@app.get("/api/gpr-stream")
async def gpr_stream_sse(lat: float = 39.9, lng: float = 32.8):
    async def event_generator():
        seed_val = int((lat + lng) * 10000) % (2**31)
        random.seed(seed_val)

        depth_points = 100
        for t in range(depth_points):
            amp = 0
            for layer in range(5):
                layer_depth = random.uniform(10, 80) * (layer + 1)
                amp += random.uniform(0.3, 0.9) * math.exp(-t * 0.5 / 200) * math.sin(2 * math.pi * t * 0.5 / (layer_depth + 10))
            amp += random.gauss(0, 0.05)
            amp = max(-1, min(1, amp))
            msg = json.dumps({
                "type": "signal",
                "time": t,
                "depth": round(t * 0.5, 1),
                "amplitude": round(amp, 4),
                "progress": round((t + 1) / depth_points * 100, 1)
            })
            yield f"data: {msg}\n\n"
            await asyncio.sleep(0.03)

        yield f"data: {json.dumps({'type': 'complete', 'message': 'Tarama tamamlandi'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no"
    })


@app.get("/")
async def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
