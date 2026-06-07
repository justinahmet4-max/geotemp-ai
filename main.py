from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import asyncio
import math
from datetime import datetime
import httpx

app = FastAPI(title="GeoTemp-AI", version="5.0.0")

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
                url = "https://deprem.afad.gov.tr/apiv2/event/filter?start=2024-01-01&end=2025-12-31&minmag=2&maxmag=8&limit=500"
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


def gaussian_blur(grid: list, grid_size: int, sigma: float) -> list:
    kernel_radius = int(math.ceil(sigma * 2))
    result = []
    for i in range(grid_size):
        for j in range(grid_size):
            val = 0.0
            weight_sum = 0.0
            for di in range(-kernel_radius, kernel_radius + 1):
                for dj in range(-kernel_radius, kernel_radius + 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < grid_size and 0 <= nj < grid_size:
                        w = math.exp(-(di * di + dj * dj) / (2 * sigma * sigma))
                        val += grid[ni * grid_size + nj] * w
                        weight_sum += w
            result.append(val / weight_sum if weight_sum > 0 else 0)
    return result


def calculate_slope_from_elev(elevations: list, grid_size: int) -> float:
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


def classify_zemin(elevation: float, slope: float) -> str:
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


ZEMIN = {
    "ZA": {"ad": "Zemin Sinifi ZA", "aciklama": "Kayali zemin", "vs30_min": 800, "vs30_max": 1500, "renk": "#2d5a27"},
    "ZB": {"ad": "Zemin Sinifi ZB", "aciklama": "Sert killi/kumlu tabaka", "vs30_min": 500, "vs30_max": 800, "renk": "#6b8e5a"},
    "ZC": {"ad": "Zemin Sinifi ZC", "aciklama": "Orta killi zemin", "vs30_min": 300, "vs30_max": 500, "renk": "#c4a882"},
    "ZD": {"ad": "Zemin Sinifi ZD", "aciklama": "Yumusak killi zemin", "vs30_min": 150, "vs30_max": 300, "renk": "#d4a574"},
    "ZE": {"ad": "Zemin Sinifi ZE", "aciklama": "Balcikli zemin", "vs30_min": 0, "vs30_max": 150, "renk": "#cc6633"},
}

TURKISH_MONTHS = {
    "January": "Ocak", "February": "Subat", "March": "Mart",
    "April": "Nisan", "May": "Mayis", "June": "Haziran",
    "July": "Temmuz", "August": "Agustos", "September": "Eylul",
    "October": "Ekim", "November": "Kasim", "December": "Aralik"
}

REGION_NAMES = [
    "Anadolu Bolgesi", "Ic Anadolu", "Bati Karadeniz",
    "Ege Kiyilari", "Akdeniz Havzasi", "Dogu Anadolu",
    "Marmara Bolgesi", "Guneydogu Anadolu"
]


def determine_region(lat: float, lng: float) -> str:
    if 40.0 <= lat <= 42.0 and 26.0 <= lng <= 30.0:
        return "Marmara Bolgesi"
    elif 40.0 <= lat <= 42.0 and 30.0 <= lng <= 37.0:
        return "Bati Karadeniz"
    elif 36.0 <= lat <= 38.0 and 26.0 <= lng <= 30.0:
        return "Ege Kiyilari"
    elif 36.0 <= lat <= 37.5 and 30.0 <= lng <= 36.0:
        return "Akdeniz Havzasi"
    elif 37.5 <= lat <= 40.0 and 30.0 <= lng <= 34.0:
        return "Ic Anadolu"
    elif 39.0 <= lat <= 41.5 and 34.0 <= lng <= 44.0:
        return "Dogu Anadolu"
    elif 36.5 <= lat <= 38.0 and 36.0 <= lng <= 44.0:
        return "Guneydogu Anadolu"
    idx = int(abs(lat * 7 + lng * 3)) % len(REGION_NAMES)
    return REGION_NAMES[idx]


async def generate_real_temperature(bounds: BoundingBox) -> dict:
    lat = (bounds.north + bounds.south) / 2
    lng = (bounds.east + bounds.west) / 2
    center_elev = await fetch_elevation(lat, lng)
    if center_elev is None:
        center_elev = 500.0

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
                month_tr = TURKISH_MONTHS.get(dt.strftime("%B"), "")
                if month_tr:
                    if month_tr not in month_temps:
                        month_temps[month_tr] = []
                    if idx < len(tmax) and idx < len(tmin):
                        avg = (tmax[idx] + tmin[idx]) / 2
                        month_temps[month_tr].append(round(avg, 1))
            except Exception:
                continue

        elev_correction = -center_elev * 0.0065

        for ay in ["Ocak", "Subat", "Mart", "Nisan", "Mayis", "Haziran",
                    "Temmuz", "Agustos", "Eylul", "Ekim", "Kasim", "Aralik"]:
            if ay in month_temps and month_temps[ay]:
                avg = sum(month_temps[ay]) / len(month_temps[ay])
                corrected = round(avg + elev_correction, 1)
                monthly_data.append({"ay": ay, "sicaklik": corrected, "kaynak": "open-meteo"})
            else:
                monthly_data.append({"ay": ay, "sicaklik": 0, "kaynak": "open-meteo"})
    else:
        for ay in ["Ocak", "Subat", "Mart", "Nisan", "Mayis", "Haziran",
                    "Temmuz", "Agustos", "Eylul", "Ekim", "Kasim", "Aralik"]:
            monthly_data.append({"ay": ay, "sicaklik": 0, "kaynak": "error"})

    return {
        "region": determine_region(lat, lng),
        "bounds": {"north": bounds.north, "south": bounds.south, "east": bounds.east, "west": bounds.west},
        "monthly_data": monthly_data,
        "kaynak": "open-meteo" if meteo else "error",
        "koordinat": {"lat": round(lat, 4), "lng": round(lng, 4)}
    }


async def generate_geological_data(bounds: BoundingBox) -> dict:
    lat = (bounds.north + bounds.south) / 2
    lng = (bounds.east + bounds.west) / 2
    center_elev = await fetch_elevation(lat, lng)
    elevations_10 = await fetch_elevation_grid(bounds, 10)
    elevations_20 = await fetch_elevation_grid(bounds, 20)

    if center_elev is None:
        center_elev = 500.0
    if elevations_10 and len(elevations_10) >= 100:
        slope = calculate_slope_from_elev(elevations_10, 10)
        avg_elev = sum(elevations_10) / len(elevations_10)
    else:
        slope = 5.0
        avg_elev = center_elev

    meteo = await fetch_open_meteo(lat, lng)
    annual_precip = 600
    if meteo and "daily" in meteo:
        precip_data = meteo["daily"].get("precipitation_sum", [])
        if precip_data:
            annual_precip = sum(p for p in precip_data if p is not None)

    ana_sinif = classify_zemin(center_elev, slope)
    vs30_mid = (ZEMIN[ana_sinif]["vs30_min"] + ZEMIN[ana_sinif]["vs30_max"]) / 2
    heyelan_riski = 0.0
    if slope > 20:
        heyelan_riski += 0.35
    elif slope > 10:
        heyelan_riski += 0.2
    elif slope > 5:
        heyelan_riski += 0.1
    if center_elev > 1000:
        heyelan_riski += 0.15
    if annual_precip > 800:
        heyelan_riski += 0.2
    elif annual_precip > 500:
        heyelan_riski += 0.1
    heyelan_riski = round(min(1.0, max(0.0, heyelan_riski)), 2)

    grid_size = 20
    zemin_grid = []
    elev_grid = elevations_20 if elevations_20 and len(elevations_20) >= grid_size * grid_size else None

    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if elev_grid and idx < len(elev_grid):
                e = elev_grid[idx]
                s = 2.0 if e < 200 else 8.0 if e < 500 else 15.0 if e < 1000 else 25.0
            else:
                e = center_elev
                s = slope

            sinif = classify_zemin(e, s)
            vs30_val = round((ZEMIN[sinif]["vs30_min"] + ZEMIN[sinif]["vs30_max"]) / 2, 0)
            zemin_grid.append({
                "sinif": sinif,
                "vs30": vs30_val,
                "renk": ZEMIN[sinif]["renk"],
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
        "sinif_bilgisi": ZEMIN[ana_sinif],
        "vs30": vs30_mid,
        "heyelan_riski": heyelan_riski,
        "aluvyon_kalinlik": round(max(2, min(50, center_elev * 0.03)), 1),
        "yeralti_su_derinligi": round(max(5, min(120, center_elev * 0.08)), 1),
        "elevation": round(center_elev, 1),
        "slope": round(slope, 1),
        "yagis_mm": round(annual_precip, 0),
        "yakin_depremler": nearby_quakes[:5],
        "kaynak": "open-elevation+afad" if elevations_10 else "error",
        "grid_size": grid_size,
        "zemin_grid": zemin_grid
    }


async def generate_spectral_data(bounds: BoundingBox) -> dict:
    lat = (bounds.north + bounds.south) / 2
    lng = (bounds.east + bounds.west) / 2
    elevations = await fetch_elevation_grid(bounds, 10)
    meteo = await fetch_open_meteo(lat, lng)

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
    surface_temp_grid = []

    for i in range(grid_size):
        for j in range(grid_size):
            lat_p = bounds.south + (bounds.north - bounds.south) * i / (grid_size - 1)
            lng_p = bounds.west + (bounds.east - bounds.west) * j / (grid_size - 1)
            lat_effect = math.sin(lat_p * 0.08) * 0.3
            ndvi = max(0, min(1, 0.2 + elev_factor * 0.3 + precip_factor * 0.3 + lat_effect))
            ndwi = max(0, min(1, 0.1 + precip_factor * 0.4 + math.cos(lng_p * 0.05) * 0.2))
            mineral = max(0, min(1, 0.4 + elev_factor * 0.3 + math.sin(lat_p * 0.15 + lng_p * 0.1) * 0.2))
            s_temp = max(0, min(1, 0.3 + (1 - elev_factor) * 0.4 + precip_factor * 0.1))
            ndvi_grid.append(round(ndvi, 3))
            ndwi_grid.append(round(ndwi, 3))
            mineral_grid.append(round(mineral, 3))
            surface_temp_grid.append(round(s_temp, 3))

    center_idx = grid_size * grid_size // 2
    return {
        "bounds": {"north": bounds.north, "south": bounds.south, "east": bounds.east, "west": bounds.west},
        "grid_size": grid_size,
        "ndvi": {"grid": ndvi_grid, "min": round(min(ndvi_grid), 3), "max": round(max(ndvi_grid), 3)},
        "ndwi": {"grid": ndwi_grid, "min": round(min(ndwi_grid), 3), "max": round(max(ndwi_grid), 3)},
        "mineral": {"grid": mineral_grid, "min": round(min(mineral_grid), 3), "max": round(max(mineral_grid), 3)},
        "surface_temp": {"grid": surface_temp_grid, "min": round(min(surface_temp_grid), 3), "max": round(max(surface_temp_grid), 3)},
        "yorum": {
            "bitki_durumu": "Iyi" if ndvi_grid[center_idx] > 0.5 else "Orta" if ndvi_grid[center_idx] > 0.25 else "Dusuk",
            "nem_durumu": "Yuksek" if ndwi_grid[center_idx] > 0.5 else "Orta" if ndwi_grid[center_idx] > 0.25 else "Dusuk",
            "mineral_bilesim": "Zengin" if mineral_grid[center_idx] > 0.6 else "Orta" if mineral_grid[center_idx] > 0.3 else "Fakir"
        },
        "kaynak": "open-elevation+meteo" if elevations else "error"
    }


def generate_gpr_profile(lat: float, lng: float) -> dict:
    seed_val = int((lat + lng) * 10000) % (2**31)
    time_window = 200
    signal = []
    depth = []
    num_layers = 5
    layer_depths = sorted([10 + (seed_val >> (i * 3)) % 35 for i in range(num_layers)])
    layer_amps = [0.3 * math.exp(-d / 80) for d in layer_depths]
    layer_widths = [2.0 + (seed_val >> (i * 5)) % 3 for i in range(num_layers)]

    for t in range(time_window):
        d = t * 0.25
        depth.append(round(d, 2))
        amp = 0.0
        for i in range(num_layers):
            ld = layer_depths[i]
            la = layer_amps[i]
            lw = layer_widths[i]
            amp += la * math.exp(-((d - ld) ** 2) / (2 * lw * lw))
        amp = max(-1, min(1, amp))
        signal.append(round(amp, 4))

    katman_names = ["Toprak Yuzey", "Killi Tabaka", "Kumlu Tabaka", "Kirec Taski", "Ana Kaya"]
    katmanlar = []
    for i in range(num_layers):
        katmanlar.append({
            "ad": katman_names[i],
            "derinlik": round(layer_depths[i], 1),
            "kalinlik": round(layer_widths[i] * 5, 1),
        })
    katmanlar.append({"ad": "Ana Kaya", "derinlik": round(layer_depths[-1] + 20, 1), "kalinlik": 0})

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
                temp = 20 - elev * 0.0065
                all_temps.append(round(temp, 1))
                grid.append(round(temp, 1))
        blurred = gaussian_blur(grid, grid_size, 2.0)
        return {
            "bounds": {"north": float(bounds.north), "south": float(bounds.south), "east": float(bounds.east), "west": float(bounds.west)},
            "grid_size": grid_size, "grid": blurred,
            "min_temp": round(min(blurred), 1), "max_temp": round(max(blurred), 1),
            "kaynak": "open-elevation"
        }
    else:
        lat_c = (bounds.north + bounds.south) / 2
        lng_c = (bounds.east + bounds.west) / 2
        base = 20 - (lat_c - 36) * 0.8 - (lng_c - 30) * 0.2
        grid = []
        all_temps = []
        for i in range(grid_size):
            for j in range(grid_size):
                lat_p = bounds.south + (bounds.north - bounds.south) * i / (grid_size - 1)
                lng_p = bounds.west + (bounds.east - bounds.west) * j / (grid_size - 1)
                grad_lat = -2 + 4 * i / (grid_size - 1)
                grad_lon = -1.5 + 3 * j / (grid_size - 1)
                temp = round(base + grad_lat + grad_lon, 1)
                all_temps.append(temp)
                grid.append(temp)
        blurred = gaussian_blur(grid, grid_size, 2.0)
        return {
            "bounds": {"north": float(bounds.north), "south": float(bounds.south), "east": float(bounds.east), "west": float(bounds.west)},
            "grid_size": grid_size, "grid": blurred,
            "min_temp": round(min(blurred), 1), "max_temp": round(max(blurred), 1),
            "kaynak": "open-elevation"
        }


@app.post("/api/geological-data")
async def geological_data(bounds: BoundingBox):
    return await generate_geological_data(bounds)


@app.post("/api/spectral-analysis")
async def spectral_analysis(bounds: BoundingBox):
    return await generate_spectral_data(bounds)


@app.post("/api/environmental-grid")
async def environmental_grid(bounds: BoundingBox):
    grid_size = 30
    elevations = await fetch_elevation_grid(bounds, grid_size)
    if not elevations or len(elevations) < grid_size * grid_size:
        elevations = [500.0] * (grid_size * grid_size)

    lat = (bounds.north + bounds.south) / 2
    lng = (bounds.east + bounds.west) / 2
    meteo = await fetch_open_meteo(lat, lng)
    annual_precip = 600
    if meteo and "daily" in meteo:
        precip_data = meteo["daily"].get("precipitation_sum", [])
        if precip_data:
            annual_precip = sum(p for p in precip_data if p is not None)

    ndvi_grid = []
    moisture_grid = []
    topsoil_grid = []
    sub_heat_grid = []
    precip_norm = min(1.0, annual_precip / 1200)

    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            e = elevations[idx]
            elev_norm = max(0, min(1, e / 2500))
            ndvi = max(0, min(1, 0.45 + precip_norm * 0.3 - elev_norm * 0.25))
            moisture = max(0, min(1, 0.3 + precip_norm * 0.4 - elev_norm * 0.2 + (0.15 if e < 100 else 0)))
            if e < 200:
                topsoil = 0.7
            elif e < 500:
                topsoil = 0.5
            elif e < 1000:
                topsoil = 0.3
            else:
                topsoil = 0.15
            heat = max(0, min(1, 0.3 + (1 - elev_norm) * 0.35 + precip_norm * 0.1))
            ndvi_grid.append(round(ndvi, 3))
            moisture_grid.append(round(moisture, 3))
            topsoil_grid.append(round(topsoil, 3))
            sub_heat_grid.append(round(heat, 3))

    ndvi_blurred = gaussian_blur(ndvi_grid, grid_size, 2.5)
    moisture_blurred = gaussian_blur(moisture_grid, grid_size, 2.5)
    topsoil_blurred = gaussian_blur(topsoil_grid, grid_size, 2.5)
    heat_blurred = gaussian_blur(sub_heat_grid, grid_size, 2.5)

    return {
        "bounds": {"north": bounds.north, "south": bounds.south, "east": bounds.east, "west": bounds.west},
        "grid_size": grid_size,
        "ndvi": {"grid": ndvi_blurred, "min": round(min(ndvi_blurred), 3), "max": round(max(ndvi_blurred), 3)},
        "moisture": {"grid": moisture_blurred, "min": round(min(moisture_blurred), 3), "max": round(max(moisture_blurred), 3)},
        "topsoil": {"grid": topsoil_blurred, "min": round(min(topsoil_blurred), 3), "max": round(max(topsoil_blurred), 3)},
        "sub_heat": {"grid": heat_blurred, "min": round(min(heat_blurred), 3), "max": round(max(heat_blurred), 3)},
        "kaynak": "open-elevation+meteo" if elevations else "error"
    }


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
            profile = generate_gpr_profile(lat, lng)
            for i, amp in enumerate(profile["signal"]):
                await websocket.send_json({
                    "type": "signal",
                    "time": i,
                    "depth": profile["depth"][i],
                    "amplitude": amp,
                    "progress": round((i + 1) / len(profile["signal"]) * 100, 1)
                })
                await asyncio.sleep(0.02)
            await websocket.send_json({"type": "complete", "message": "Tarama tamamlandi"})
    except WebSocketDisconnect:
        pass


@app.get("/api/gpr-stream")
async def gpr_stream_sse(lat: float = 39.9, lng: float = 32.8):
    async def event_generator():
        profile = generate_gpr_profile(lat, lng)
        for i, amp in enumerate(profile["signal"]):
            msg = json.dumps({
                "type": "signal",
                "time": i,
                "depth": profile["depth"][i],
                "amplitude": amp,
                "progress": round((i + 1) / len(profile["signal"]) * 100, 1)
            })
            yield f"data: {msg}\n\n"
            await asyncio.sleep(0.02)
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
