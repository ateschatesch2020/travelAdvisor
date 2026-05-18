import logging
from langchain_core.tools import tool
import os
from dotenv import load_dotenv
load_dotenv()
import serpapi

logger = logging.getLogger(__name__)


def _extract_flights(raw):
    all_flights = []
    for group in raw.get("best_flights", []) + raw.get("other_flights", []):
        legs = group.get("flights", [])
        if not legs:
            continue
        stops = len(legs) - 1
        layover_names = [l.get("name", "") for l in group.get("layovers", [])]
        baggage_keywords = ("bag", "kg", "lb", "luggage", "allowance")
        extensions = group.get("extensions", [])
        baggage = [e for e in extensions if any(kw in e.lower() for kw in baggage_keywords)]
        for leg in legs:
            for e in leg.get("extensions", []):
                if any(kw in e.lower() for kw in baggage_keywords) and e not in baggage:
                    baggage.append(e)
        first, last = legs[0], legs[-1]
        all_flights.append({
            "airline": first.get("airline"),
            "flight_number": first.get("flight_number"),
            "departure": first.get("departure_airport", {}).get("time"),
            "arrival": last.get("arrival_airport", {}).get("time"),
            "duration_min": group.get("total_duration"),
            "stops": stops,
            "layovers": layover_names,
            "price_usd": group.get("price"),
            "baggage": baggage,
        })
    return all_flights


def _get_image_url(images):
    if not images:
        return None
    url = images[0].get("thumbnail") or images[0].get("original_image")
    return url if url and url.startswith("https://") else None


def _extract_hotels(raw):
    hotels = []
    for prop in raw.get("properties", []):
        rate = prop.get("rate_per_night", {})
        images = prop.get("images", [])
        hotel = {
            "name": prop.get("name"),
            "type": prop.get("type"),
            "location": prop.get("neighborhood") or prop.get("address"),
            "price_per_night": rate.get("lowest"),
            "free_cancellation_until": prop.get("free_cancellation_until"),
            "check_in_time": prop.get("check_in_time"),
            "check_out_time": prop.get("check_out_time"),
            "amenities": prop.get("amenities", []),
            "nearby_places": [p.get("name") for p in prop.get("nearby_places", [])],
        }
        image_url = _get_image_url(images)
        if image_url:
            hotel["image"] = image_url
        hotels.append(hotel)
    return hotels


@tool
def search_hotels(location: str, check_in_date: str, check_out_date: str = "") -> list:
    """Search for real-time hotel data using Google Hotels. Use this tool for ANY hotel availability, price, or recommendation question.
    location: city or area name (e.g. 'Madrid', 'Paris city center')
    check_in_date: check-in date in YYYY-MM-DD format (e.g. '2026-05-27')
    check_out_date: check-out date in YYYY-MM-DD format. If not provided, defaults to the next day.
    Always convert dates to YYYY-MM-DD format before calling this tool.
    Returns name, type, location, price per night, free cancellation info, amenities, nearby places and image.
    """
    from datetime import datetime, timedelta
    if not check_in_date:
        return "check_in_date is required. Please ask the user for a specific check-in date in YYYY-MM-DD format (e.g., '2026-06-15')."
    if not check_out_date:
        check_out_date = (datetime.strptime(check_in_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    check_in_dt  = datetime.strptime(check_in_date,  "%Y-%m-%d")
    check_out_dt = datetime.strptime(check_out_date, "%Y-%m-%d")
    if (check_out_dt - check_in_dt).days > 7:
        return "Hotel stay exceeds 7 days. Please limit hotel searches to a maximum of 7 nights."
    client = serpapi.Client(api_key=os.getenv("SERPAPI_KEY"))
    raw = client.search({
        "engine": "google_hotels",
        "q": f"Hotels in {location}",
        "check_in_date": check_in_date,
        "check_out_date": check_out_date,
        "hl": "en",
        "currency": "EUR",
    })
    return _extract_hotels(raw)


@tool
def search_flights(departure_id: str, arrival_id: str, outbound_date: str) -> list:
    """Search for real-time one-way flight data using Google Flights. Use this tool for ANY flight availability or price question.
    departure_id: IATA airport code (e.g. 'MUC' for Munich)
    arrival_id: IATA airport code (e.g. 'MAD' for Madrid)
    outbound_date: date in YYYY-MM-DD format (e.g. '2026-05-23')
    Returns a list of flights with airline, times, stops, price and baggage info.
    """
    client = serpapi.Client(api_key=os.getenv("SERPAPI_KEY"))
    raw = client.search({
        "engine": "google_flights",
        "departure_id": departure_id,
        "arrival_id": arrival_id,
        "outbound_date": outbound_date,
        "type": "2",
    })
    return _extract_flights(raw)


@tool
def search_weather(city: str, date: str) -> dict:
    """Search for weather forecast using WeatherAPI. Use this tool for ANY weather, temperature, rain, or wind question.
    city: city name (e.g. 'Madrid', 'Munich')
    date: date in YYYY-MM-DD format (e.g. '2026-05-27'). Must be within 14 days from today.
    Returns temperature (min/max/avg in Celsius), precipitation chance, total rain in mm, and max wind speed.
    """
    import requests
    url = "http://api.weatherapi.com/v1/forecast.json"
    params = {"q": city, "dt": date, "days": 1}
    headers = {"key": os.getenv("WEATHER_API_KEY")}
    raw = requests.get(url, params=params, headers=headers).json()
    forecastdays = raw.get("forecast", {}).get("forecastday", [])
    if not forecastdays:
        logger.warning("search_weather: no forecast data for %s on %s", city, date)
        return {}
    day = forecastdays[0]["day"]
    return {
        "city": raw["location"]["name"],
        "date": date,
        "temp_min_c": day["mintemp_c"],
        "temp_max_c": day["maxtemp_c"],
        "temp_avg_c": day["avgtemp_c"],
        "rain_mm": day["totalprecip_mm"],
        "rain_chance_pct": day["daily_chance_of_rain"],
        "max_wind_kph": day["maxwind_kph"],
        "condition": day["condition"]["text"],
    }


class Tools:
    tools = [search_flights, search_hotels, search_weather]
