from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from ortools.sat.python import cp_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WeatherConstraints:
    min_temp_c: float = -999.0
    max_temp_c: float = 999.0
    max_rain_pct: float = 100.0
    max_wind_kph: float = 999.0


@dataclass
class ItineraryInput:
    candidate_cities: list[str]
    city_iata: dict[str, str]
    origin_city: str
    origin_iata: str
    num_to_visit: int
    start_date: date
    end_date: date
    min_nights: int
    max_nights: int
    flights_data: dict   # (from_iata, to_iata, date_str) -> list[flight_dict]
    weather_data: dict   # (city_name, date_str) -> weather_dict
    weather_constraints: WeatherConstraints = field(default_factory=WeatherConstraints)


@dataclass
class LegOption:
    pos: int
    from_city_idx: int   # index into candidate_cities; -1 means origin
    to_city_idx: int     # index into candidate_cities; -1 means origin
    date_offset: int     # days from start_date
    price_cents: int
    flight: dict


@dataclass
class Itinerary:
    legs: list[LegOption]
    total_cost_usd: float
    city_order: list[str]
    nights_per_city: list[int]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_weather(
    weather_data: dict,
    city_name: str,
    date_str: str,
    constraints: WeatherConstraints,
) -> bool:
    w = weather_data.get((city_name, date_str))
    if w is None:
        return True
    return (
        w.get("temp_max_c", 999) >= constraints.min_temp_c
        and w.get("temp_min_c", -999) <= constraints.max_temp_c
        and w.get("rain_chance_pct", 0) <= constraints.max_rain_pct
        and w.get("max_wind_kph", 0) <= constraints.max_wind_kph
    )


def _date_str(base: date, offset: int) -> str:
    return (base + timedelta(days=offset)).strftime("%Y-%m-%d")


def _build_leg_options(inp: ItineraryInput) -> dict[int, list[LegOption]]:
    k = inp.num_to_visit
    N = len(inp.candidate_cities)
    total_days = (inp.end_date - inp.start_date).days
    idx = {city: i for i, city in enumerate(inp.candidate_cities)}

    options: dict[int, list[LegOption]] = {pos: [] for pos in range(k + 1)}

    def add_options(pos: int, from_iata: str, from_idx: int, to_city: str, to_idx: int, day_range):
        to_iata = inp.city_iata[to_city] if to_idx >= 0 else inp.origin_iata
        for day in day_range:
            ds = _date_str(inp.start_date, day)
            flights = inp.flights_data.get((from_iata, to_iata, ds), [])
            for fl in flights:
                price = fl.get("price_usd")
                if price is None:
                    continue
                if to_idx >= 0 and not _check_weather(
                    inp.weather_data, to_city, ds, inp.weather_constraints
                ):
                    continue
                options[pos].append(LegOption(
                    pos=pos,
                    from_city_idx=from_idx,
                    to_city_idx=to_idx,
                    date_offset=day,
                    price_cents=int(round(price * 100)),
                    flight=fl,
                ))

    outbound_days = range(0, max(0, total_days - k * inp.min_nights) + 1)
    return_days   = range(k * inp.min_nights, total_days + 1)
    internal_days = range(inp.min_nights, max(inp.min_nights, total_days - inp.min_nights) + 1)

    # Leg 0: origin → each candidate city
    for city in inp.candidate_cities:
        add_options(0, inp.origin_iata, -1, city, idx[city], outbound_days)

    # Legs 1..k-1: any candidate city → any different candidate city
    for pos in range(1, k):
        for from_city in inp.candidate_cities:
            from_iata = inp.city_iata[from_city]
            for to_city in inp.candidate_cities:
                if from_city == to_city:
                    continue
                add_options(pos, from_iata, idx[from_city], to_city, idx[to_city], internal_days)

    # Leg k: each candidate city → origin
    for from_city in inp.candidate_cities:
        from_iata = inp.city_iata[from_city]
        add_options(k, from_iata, idx[from_city], inp.origin_city, -1, return_days)

    return options


# ---------------------------------------------------------------------------
# CP-SAT solver
# ---------------------------------------------------------------------------

def solve_itinerary(inp: ItineraryInput) -> Optional[Itinerary]:
    if not (1 <= inp.num_to_visit <= len(inp.candidate_cities)):
        raise ValueError("num_to_visit must be between 1 and len(candidate_cities)")

    options = _build_leg_options(inp)
    k = inp.num_to_visit
    N = len(inp.candidate_cities)
    total_days = (inp.end_date - inp.start_date).days

    for pos in range(k + 1):
        if not options[pos]:
            return None

    model = cp_model.CpModel()

    # x[pos][i]: option i chosen at position pos
    x: dict[int, list] = {
        pos: [model.new_bool_var(f"x_{pos}_{i}") for i in range(len(options[pos]))]
        for pos in range(k + 1)
    }

    # city_id[pos]: which candidate city (0..N-1) is at visiting position pos
    city_id = [model.new_int_var(0, N - 1, f"city_{pos}") for pos in range(k)]

    # dep_day[pos]: departure day offset
    dep_day = [model.new_int_var(0, total_days, f"dep_{pos}") for pos in range(k + 1)]

    # 1. Exactly one option per leg
    for pos in range(k + 1):
        model.add_exactly_one(x[pos])

    # 2 & 3. Link flight selection → city_id and dep_day
    for pos in range(k):
        for i, opt in enumerate(options[pos]):
            model.add(city_id[pos] == opt.to_city_idx).only_enforce_if(x[pos][i])
            model.add(dep_day[pos] == opt.date_offset).only_enforce_if(x[pos][i])

    for i, opt in enumerate(options[k]):
        model.add(dep_day[k] == opt.date_offset).only_enforce_if(x[k][i])

    # 4. Connectivity: to_city of leg pos == from_city of leg pos+1
    for pos in range(k - 1):
        for j, opt_next in enumerate(options[pos + 1]):
            model.add(city_id[pos] == opt_next.from_city_idx).only_enforce_if(x[pos + 1][j])

    # Return leg must depart from the last visited city
    for j, opt_ret in enumerate(options[k]):
        model.add(city_id[k - 1] == opt_ret.from_city_idx).only_enforce_if(x[k][j])

    # 5. All visited cities are distinct
    if k > 1:
        model.add_all_different(city_id)

    # 6 & 7. Nights per city: min_nights ≤ gap ≤ max_nights
    for pos in range(k):
        model.add(dep_day[pos + 1] - dep_day[pos] >= inp.min_nights)
        model.add(dep_day[pos + 1] - dep_day[pos] <= inp.max_nights)

    # 8. Vacation window bounds
    model.add(dep_day[0] >= 0)
    model.add(dep_day[k] <= total_days)

    # Objective: minimize total flight cost
    model.minimize(
        sum(opt.price_cents * x[pos][i]
            for pos in range(k + 1)
            for i, opt in enumerate(options[pos]))
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    solver.parameters.num_search_workers = 4
    status = solver.solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    chosen_legs: list[LegOption] = []
    for pos in range(k + 1):
        for i, opt in enumerate(options[pos]):
            if solver.value(x[pos][i]):
                chosen_legs.append(opt)
                break

    city_order = [inp.candidate_cities[leg.to_city_idx] for leg in chosen_legs[:k]]
    nights = [
        chosen_legs[p + 1].date_offset - chosen_legs[p].date_offset
        for p in range(k)
    ]

    return Itinerary(
        legs=chosen_legs,
        total_cost_usd=solver.objective_value / 100.0,
        city_order=city_order,
        nights_per_city=nights,
    )


# ---------------------------------------------------------------------------
# Data fetching (calls tools.py)
# ---------------------------------------------------------------------------

def _fetch_flights(
    candidate_cities: list[str],
    city_iata: dict[str, str],
    origin_iata: str,
    start_date: date,
    end_date: date,
    min_nights: int,
    num_to_visit: int,
) -> tuple[dict, int]:
    from tools import search_flights

    k = num_to_visit
    total_days = (end_date - start_date).days
    flights_data: dict = {}
    routes_searched = 0

    candidate_iatas = [city_iata[c] for c in candidate_cities]

    routes = []
    # Outbound: origin → each candidate
    outbound_end = max(0, total_days - k * min_nights)
    for iata in candidate_iatas:
        routes.append((origin_iata, iata, range(outbound_end + 1)))

    # Internal: candidate → different candidate
    internal_start = min_nights
    internal_end = max(min_nights, total_days - min_nights)
    for a in candidate_iatas:
        for b in candidate_iatas:
            if a != b:
                routes.append((a, b, range(internal_start, internal_end + 1)))

    # Return: each candidate → origin
    return_start = k * min_nights
    for iata in candidate_iatas:
        routes.append((iata, origin_iata, range(return_start, total_days + 1)))

    for from_iata, to_iata, day_range in routes:
        for day in day_range:
            ds = _date_str(start_date, day)
            key = (from_iata, to_iata, ds)
            if key in flights_data:
                continue
            routes_searched += 1
            try:
                result = search_flights.invoke(
                    {"departure_id": from_iata, "arrival_id": to_iata, "outbound_date": ds}
                )
                if result:
                    flights_data[key] = result
            except Exception:
                logger.warning("search_flights failed for %s→%s on %s", from_iata, to_iata, ds, exc_info=True)

    return flights_data, routes_searched


def _fetch_weather(
    candidate_cities: list[str],
    start_date: date,
    end_date: date,
) -> dict:
    from tools import search_weather

    weather_data: dict = {}
    total_days = (end_date - start_date).days

    for city in candidate_cities:
        for day in range(total_days + 1):
            ds = _date_str(start_date, day)
            key = (city, ds)
            if key in weather_data:
                continue
            try:
                result = search_weather.invoke({"city": city, "date": ds})
                if result:
                    weather_data[key] = result
            except Exception:
                logger.warning("search_weather failed for %s on %s", city, ds, exc_info=True)

    return weather_data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def optimize_itinerary(
    candidate_cities: list[str],
    city_iata: dict[str, str],
    origin_city: str,
    origin_iata: str,
    num_to_visit: int,
    start_date: date,
    end_date: date,
    min_nights: int,
    max_nights: int,
    weather_constraints: WeatherConstraints = WeatherConstraints(),
) -> tuple[Optional[Itinerary], dict]:
    """Find the cheapest multi-city itinerary using live flight and weather data.

    Fetches prices from search_flights and weather from search_weather (tools.py),
    then runs a CP-SAT optimizer to select the lowest-cost feasible itinerary.

    Args:
        candidate_cities: Pool of city names to choose from.
        city_iata: Mapping of city name to IATA airport code.
        origin_city: Home city name.
        origin_iata: Home city IATA code.
        num_to_visit: How many cities to select and visit (k).
        start_date: Earliest departure date from home.
        end_date: Latest return date to home.
        min_nights: Minimum nights to spend in each city.
        max_nights: Maximum nights to spend in each city.
        weather_constraints: Optional weather thresholds.

    Returns:
        Itinerary with optimal city order, flights, dates, and total cost,
        or None if no feasible itinerary exists.
    """
    logger.info("Fetching flight data for %d candidate cities...", len(candidate_cities))
    flights_data, routes_searched = _fetch_flights(
        candidate_cities, city_iata, origin_iata,
        start_date, end_date, min_nights, num_to_visit,
    )
    logger.info("%d route/date combinations found.", len(flights_data))

    logger.info("Fetching weather data...")
    weather_data = _fetch_weather(candidate_cities, start_date, end_date)
    logger.info("%d city/date weather records found.", len(weather_data))

    stats = {
        "routes_searched": routes_searched,
        "routes_with_flights": len(flights_data),
        "total_flight_options": sum(len(v) for v in flights_data.values()),
        "weather_records": len(weather_data),
    }

    inp = ItineraryInput(
        candidate_cities=candidate_cities,
        city_iata=city_iata,
        origin_city=origin_city,
        origin_iata=origin_iata,
        num_to_visit=num_to_visit,
        start_date=start_date,
        end_date=end_date,
        min_nights=min_nights,
        max_nights=max_nights,
        flights_data=flights_data,
        weather_data=weather_data,
        weather_constraints=weather_constraints,
    )

    logger.info("Running CP-SAT optimizer...")
    return solve_itinerary(inp), stats


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verify(inp: ItineraryInput, result: Itinerary) -> list[str]:
    k = inp.num_to_visit
    errors: list[str] = []

    if len(result.legs) != k + 1:
        errors.append(f"Expected {k+1} legs, got {len(result.legs)}")
        return errors

    for p in range(k):
        cur, nxt = result.legs[p], result.legs[p + 1]
        if cur.to_city_idx != nxt.from_city_idx:
            errors.append(f"Leg {p} lands at idx {cur.to_city_idx} but leg {p+1} departs from idx {nxt.from_city_idx}")

    if result.legs[0].from_city_idx != -1:
        errors.append("First leg does not depart from origin")
    if result.legs[k].to_city_idx != -1:
        errors.append("Last leg does not return to origin")

    visited = [leg.to_city_idx for leg in result.legs[:k]]
    if len(set(visited)) != k:
        errors.append(f"Cities not all distinct: {visited}")

    for i, nights in enumerate(result.nights_per_city):
        if not (inp.min_nights <= nights <= inp.max_nights):
            errors.append(f"City {result.city_order[i]}: {nights} nights outside [{inp.min_nights}, {inp.max_nights}]")

    total_days = (inp.end_date - inp.start_date).days
    if result.legs[0].date_offset < 0:
        errors.append("First departure before start_date")
    if result.legs[k].date_offset > total_days:
        errors.append("Return departure after end_date")

    leg_sum_cents = sum(leg.price_cents for leg in result.legs)
    reported_cents = round(result.total_cost_usd * 100)
    if abs(leg_sum_cents - reported_cents) > 1:
        errors.append(f"Cost mismatch: legs sum {leg_sum_cents}¢, reported {reported_cents}¢")

    return errors


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    from datetime import date

    parser = argparse.ArgumentParser(
        description="Find the cheapest multi-city itinerary using CP-SAT optimization."
    )
    parser.add_argument(
        "--cities", required=True,
        help='Comma-separated candidate city names, e.g. "Madrid,Lisbon,Barcelona,Rome"',
    )
    parser.add_argument(
        "--city-iata", required=True, dest="city_iata",
        help='Comma-separated City:IATA pairs, e.g. "Madrid:MAD,Lisbon:LIS,Barcelona:BCN,Rome:FCO"',
    )
    parser.add_argument("--origin",      required=True, help="Origin city name, e.g. Munich")
    parser.add_argument("--origin-iata", required=True, dest="origin_iata", help="Origin IATA code, e.g. MUC")
    parser.add_argument("--num-to-visit", required=True, dest="num_to_visit", type=int,
                        help="How many cities to select and visit")
    parser.add_argument("--start-date",  required=True, dest="start_date",
                        help="Earliest departure date, YYYY-MM-DD")
    parser.add_argument("--end-date",    required=True, dest="end_date",
                        help="Latest return date, YYYY-MM-DD")
    parser.add_argument("--min-nights",  default=2, dest="min_nights", type=int,
                        help="Minimum nights per city (default: 2)")
    parser.add_argument("--max-nights",  default=5, dest="max_nights", type=int,
                        help="Maximum nights per city (default: 5)")
    parser.add_argument("--min-temp-c",  default=-999.0, dest="min_temp_c", type=float,
                        help="Minimum acceptable average temperature in °C")
    parser.add_argument("--max-rain-pct", default=100.0, dest="max_rain_pct", type=float,
                        help="Maximum acceptable rain chance in %% (0-100)")
    parser.add_argument("--max-wind-kph", default=999.0, dest="max_wind_kph", type=float,
                        help="Maximum acceptable wind speed in km/h")

    args = parser.parse_args()

    candidate_cities = [c.strip() for c in args.cities.split(",")]
    city_iata = dict(pair.strip().split(":") for pair in args.city_iata.split(","))
    start_date = date.fromisoformat(args.start_date)
    end_date   = date.fromisoformat(args.end_date)
    constraints = WeatherConstraints(
        min_temp_c=args.min_temp_c,
        max_rain_pct=args.max_rain_pct,
        max_wind_kph=args.max_wind_kph,
    )

    result, stats = optimize_itinerary(
        candidate_cities=candidate_cities,
        city_iata=city_iata,
        origin_city=args.origin,
        origin_iata=args.origin_iata,
        num_to_visit=args.num_to_visit,
        start_date=start_date,
        end_date=end_date,
        min_nights=args.min_nights,
        max_nights=args.max_nights,
        weather_constraints=constraints,
    )

    print(f"Search summary: {stats['routes_searched']} routes queried, "
          f"{stats['routes_with_flights']} with results "
          f"({stats['total_flight_options']} options) | "
          f"{stats['weather_records']} weather records")

    if result is None:
        print("\nNo feasible itinerary found.")
        return

    print(f"\nOptimal itinerary:")
    print(f"  City order : {' -> '.join([args.origin] + result.city_order + [args.origin])}")
    print(f"  Total cost : ${result.total_cost_usd:.2f}")
    print()

    for leg in result.legs:
        dep = (start_date + timedelta(days=leg.date_offset)).strftime("%Y-%m-%d")
        fl  = leg.flight
        print(f"  {dep}  {fl.get('airline', '?'):12s} {fl.get('flight_number', '?'):8s}  "
              f"${fl.get('price_usd', 0):.2f}")

    print()
    for city, nights in zip(result.city_order, result.nights_per_city):
        print(f"  {city}: {nights} night{'s' if nights != 1 else ''}")


if __name__ == "__main__":
    main()
