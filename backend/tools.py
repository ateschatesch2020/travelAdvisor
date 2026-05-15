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


def _extract_hotels(raw):
    hotels = []
    for prop in raw.get("properties", []):
        rate = prop.get("rate_per_night", {})
        images = prop.get("images", [])
        hotels.append({
            "name": prop.get("name"),
            "type": prop.get("type"),
            "location": prop.get("neighborhood") or prop.get("address"),
            "price_per_night": rate.get("lowest"),
            "free_cancellation_until": prop.get("free_cancellation_until"),
            "check_in_time": prop.get("check_in_time"),
            "check_out_time": prop.get("check_out_time"),
            "amenities": prop.get("amenities", []),
            "nearby_places": [p.get("name") for p in prop.get("nearby_places", [])],
            "image": images[0].get("thumbnail") if images else None,
        })
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
    if not check_out_date:
        check_out_date = (datetime.strptime(check_in_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
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


@tool
def optimize_itinerary(
    candidate_cities: str,
    city_iata_pairs: str,
    origin_city: str,
    origin_iata: str,
    num_to_visit: int,
    start_date: str,
    end_date: str,
    min_nights: int = 2,
    max_nights: int = 5,
    min_temp_c: float = -999.0,
    max_rain_pct: float = 100.0,
) -> str:
    """Find the cheapest multi-city vacation itinerary using CP-SAT optimization.
    Use this when the user wants to plan a multi-city trip, find the cheapest route
    between several cities, or optimize a vacation itinerary within a date window.
    candidate_cities: comma-separated city names to consider, e.g. 'Madrid,Lisbon,Barcelona,Rome'
    city_iata_pairs: comma-separated City:IATA pairs, e.g. 'Madrid:MAD,Lisbon:LIS,Barcelona:BCN,Rome:FCO'
    origin_city: home city name, e.g. 'Munich'
    origin_iata: home city IATA airport code, e.g. 'MUC'
    num_to_visit: how many cities to select and visit from the candidates
    start_date: earliest departure date in YYYY-MM-DD format
    end_date: latest return date in YYYY-MM-DD format
    min_nights: minimum nights per city (default 2)
    max_nights: maximum nights per city (default 5)
    min_temp_c: minimum acceptable average temperature in Celsius (default: no constraint)
    max_rain_pct: maximum acceptable rain chance in % 0-100 (default: no constraint)
    Returns the optimal city order, per-leg flight details, and total cost.
    """
    from datetime import date as _date, timedelta
    from itinerary_optimizer import (
        optimize_itinerary as _optimize,
        WeatherConstraints,
    )

    try:
        cities = [c.strip() for c in candidate_cities.split(",")]
        iata = dict(p.strip().split(":") for p in city_iata_pairs.split(","))
        start_date_obj = _date.fromisoformat(start_date)
        end_date_obj   = _date.fromisoformat(end_date)
    except Exception as e:
        logger.error("optimize_itinerary: parameter parsing failed — %s", e, exc_info=True)
        return (
            f"Parameter error: {e}. "
            "Expected format — cities: 'City1,City2', "
            "iata: 'City1:IATA1,City2:IATA2', dates: YYYY-MM-DD."
        )

    result, stats = _optimize(
        candidate_cities=cities,
        city_iata=iata,
        origin_city=origin_city,
        origin_iata=origin_iata,
        num_to_visit=num_to_visit,
        start_date=start_date_obj,
        end_date=end_date_obj,
        min_nights=min_nights,
        max_nights=max_nights,
        weather_constraints=WeatherConstraints(
            min_temp_c=min_temp_c,
            max_rain_pct=max_rain_pct,
        ),
    )

    summary = (
        f"Search summary: {stats['routes_searched']} flight routes queried, "
        f"{stats['routes_with_flights']} returned results "
        f"({stats['total_flight_options']} options total) | "
        f"{stats['weather_records']} weather records fetched."
    )

    if result is None:
        return f"{summary}\nNo feasible itinerary found within the given constraints."

    base = _date.fromisoformat(start_date)
    lines = [
        summary,
        "",
        f"Optimal route: {' -> '.join([origin_city] + result.city_order + [origin_city])}",
        f"Total flight cost: ${result.total_cost_usd:.2f}",
        "",
    ]
    for i, leg in enumerate(result.legs):
        dep = (base + timedelta(days=leg.date_offset)).strftime("%Y-%m-%d")
        fl = leg.flight
        dest = result.city_order[i] if i < len(result.city_order) else origin_city
        nights_str = f" — stay {result.nights_per_city[i]} nights" if i < len(result.nights_per_city) else ""
        lines.append(
            f"  {dep}  -> {dest}  {fl.get('airline', '')} {fl.get('flight_number', '')}  "
            f"${fl.get('price_usd', 0):.2f}{nights_str}"
        )
    return "\n".join(lines)


class Tools:
    tools = [search_flights, search_hotels, search_weather, optimize_itinerary]
