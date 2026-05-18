"""Unit tests for the optimize_itinerary CP-SAT solver and @tool wrapper.

search_flights and search_weather are mocked in all tool-level tests so no
live API calls are made.

Run with:
    python -m unittest test_optimize_itinerary
"""

import unittest
from unittest.mock import patch
from datetime import date

from itinerary_optimizer import (
    ItineraryInput,
    WeatherConstraints,
    solve_itinerary,
    _verify,
)


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _fl(price, airline="AirTest", number="AT001"):
    return {
        "airline": airline,
        "flight_number": number,
        "departure": "08:00",
        "arrival": "10:00",
        "duration_min": 120,
        "stops": 0,
        "layovers": [],
        "price_usd": price,
        "baggage": [],
    }


def _wx(city, date_str, temp=22.0, rain=10.0, wind=15.0):
    return {
        "city": city,
        "date": date_str,
        "temp_min_c": temp - 5,
        "temp_max_c": temp + 5,
        "temp_avg_c": temp,
        "rain_mm": 0.0,
        "rain_chance_pct": rain,
        "max_wind_kph": wind,
        "condition": "Sunny",
    }


# ---------------------------------------------------------------------------
# Two-city fixtures (visit both Barcelona and Madrid)
#
# Cheapest ordering: MUC→BCN $100, BCN→MAD $80, MAD→MUC $90  = $270
# Costlier ordering: MUC→MAD $150, MAD→BCN $70, BCN→MUC $120 = $340
# ---------------------------------------------------------------------------

TWO_CITY_FLIGHTS = {
    ("MUC", "BCN", "2026-06-01"): [_fl(100, "Lufthansa", "LH1000")],
    ("MUC", "MAD", "2026-06-01"): [_fl(150, "Iberia",    "IB2000")],
    ("BCN", "MAD", "2026-06-03"): [_fl(80,  "Vueling",   "VY100")],
    ("MAD", "BCN", "2026-06-03"): [_fl(70,  "Vueling",   "VY200")],
    ("MAD", "MUC", "2026-06-05"): [_fl(90,  "Iberia",    "IB3000")],
    ("BCN", "MUC", "2026-06-05"): [_fl(120, "Vueling",   "VY300")],
}

TWO_CITY_WEATHER = {
    (city, f"2026-06-0{d}"): _wx(city, f"2026-06-0{d}")
    for city in ["Barcelona", "Madrid"]
    for d in range(1, 6)
}


def _two_city_inp(
    flights=None,
    weather=None,
    constraints=None,
    num_to_visit=2,
    min_nights=2,
    max_nights=3,
):
    return ItineraryInput(
        candidate_cities=["Barcelona", "Madrid"],
        city_iata={"Barcelona": "BCN", "Madrid": "MAD"},
        origin_city="Munich",
        origin_iata="MUC",
        num_to_visit=num_to_visit,
        start_date=date(2026, 6, 1),
        end_date=date(2026, 6, 10),
        min_nights=min_nights,
        max_nights=max_nights,
        flights_data=flights if flights is not None else TWO_CITY_FLIGHTS,
        weather_data=weather if weather is not None else TWO_CITY_WEATHER,
        weather_constraints=constraints if constraints is not None else WeatherConstraints(),
    )


# ---------------------------------------------------------------------------
# Three-city fixtures (pick 2 of 3: Barcelona, Madrid, Lisbon)
#
# All weather OK — cheapest pair includes Barcelona:
#   MUC→BCN $50, BCN→MAD $30, MAD→MUC $90 = $170
#
# Barcelona filtered by weather — cheapest is Madrid→Lisbon:
#   MUC→MAD $100, MAD→LIS $60, LIS→MUC $80 = $240
# ---------------------------------------------------------------------------

THREE_CITY_FLIGHTS = {
    ("MUC", "BCN", "2026-06-01"): [_fl(50,  "AirA", "A1")],
    ("MUC", "MAD", "2026-06-01"): [_fl(100, "AirB", "B1")],
    ("MUC", "LIS", "2026-06-01"): [_fl(120, "AirC", "C1")],
    ("BCN", "MAD", "2026-06-03"): [_fl(30,  "AirD", "D1")],
    ("BCN", "LIS", "2026-06-03"): [_fl(40,  "AirE", "E1")],
    ("MAD", "BCN", "2026-06-03"): [_fl(50,  "AirF", "F1")],
    ("MAD", "LIS", "2026-06-03"): [_fl(60,  "AirG", "G1")],
    ("LIS", "BCN", "2026-06-03"): [_fl(55,  "AirH", "H1")],
    ("LIS", "MAD", "2026-06-03"): [_fl(70,  "AirI", "I1")],
    ("BCN", "MUC", "2026-06-05"): [_fl(60,  "AirJ", "J1")],
    ("MAD", "MUC", "2026-06-05"): [_fl(90,  "AirK", "K1")],
    ("LIS", "MUC", "2026-06-05"): [_fl(80,  "AirL", "L1")],
}

THREE_CITY_WEATHER_OK = {
    (city, f"2026-06-0{d}"): _wx(city, f"2026-06-0{d}")
    for city in ["Barcelona", "Madrid", "Lisbon"]
    for d in range(1, 6)
}

THREE_CITY_WEATHER_BAD_BCN = {
    (city, f"2026-06-0{d}"): _wx(
        city, f"2026-06-0{d}",
        rain=95.0 if city == "Barcelona" else 10.0,
    )
    for city in ["Barcelona", "Madrid", "Lisbon"]
    for d in range(1, 6)
}


def _three_city_inp(weather=None, constraints=None, num_to_visit=2):
    return ItineraryInput(
        candidate_cities=["Barcelona", "Madrid", "Lisbon"],
        city_iata={"Barcelona": "BCN", "Madrid": "MAD", "Lisbon": "LIS"},
        origin_city="Munich",
        origin_iata="MUC",
        num_to_visit=num_to_visit,
        start_date=date(2026, 6, 1),
        end_date=date(2026, 6, 10),
        min_nights=2,
        max_nights=3,
        flights_data=THREE_CITY_FLIGHTS,
        weather_data=weather if weather is not None else THREE_CITY_WEATHER_OK,
        weather_constraints=constraints if constraints is not None else WeatherConstraints(),
    )


# ---------------------------------------------------------------------------
# Mock side-effects used by tool-level tests
# ---------------------------------------------------------------------------

def _mock_flights(args):
    key = (args["departure_id"], args["arrival_id"], args["outbound_date"])
    return TWO_CITY_FLIGHTS.get(key, [])


def _mock_weather(args):
    return TWO_CITY_WEATHER.get((args["city"], args["date"]), {})


# ---------------------------------------------------------------------------
# Solver tests  (pure CP-SAT, no API calls)
# ---------------------------------------------------------------------------

class TestSolveItinerary(unittest.TestCase):

    def test_finds_cheapest_ordering(self):
        result, _ = solve_itinerary(_two_city_inp())
        self.assertIsNotNone(result)
        self.assertEqual(result.city_order, ["Barcelona", "Madrid"])
        self.assertAlmostEqual(result.total_cost_usd, 270.0, places=2)

    def test_solution_passes_verify(self):
        inp = _two_city_inp()
        result, _ = solve_itinerary(inp)
        self.assertIsNotNone(result)
        self.assertEqual(_verify(inp, result), [])

    def test_leg_count_equals_num_to_visit_plus_one(self):
        result, _ = solve_itinerary(_two_city_inp(num_to_visit=2))
        self.assertIsNotNone(result)
        self.assertEqual(len(result.legs), 3)

    def test_nights_within_min_max_bounds(self):
        result, _ = solve_itinerary(_two_city_inp(min_nights=2, max_nights=3))
        self.assertIsNotNone(result)
        for nights in result.nights_per_city:
            self.assertGreaterEqual(nights, 2)
            self.assertLessEqual(nights, 3)

    def test_no_flights_returns_none(self):
        result, _ = solve_itinerary(_two_city_inp(flights={}))
        self.assertIsNone(result)

    def test_impossible_window_returns_none(self):
        # Window too short: 3 days, min_nights=2, num_to_visit=2 → needs 4 days minimum
        result = solve_itinerary(_two_city_inp(
            min_nights=2,
            max_nights=2,
        ))
        # flights exist only on day 0, 2, 4 — day 4 is within the 9-day window, so this succeeds
        # To force infeasibility, shrink end_date so day 4 is out of range
        from itinerary_optimizer import ItineraryInput, WeatherConstraints
        inp = ItineraryInput(
            candidate_cities=["Barcelona", "Madrid"],
            city_iata={"Barcelona": "BCN", "Madrid": "MAD"},
            origin_city="Munich", origin_iata="MUC",
            num_to_visit=2,
            start_date=date(2026, 6, 1),
            end_date=date(2026, 6, 3),   # only 2 days — return on day 4 impossible
            min_nights=2, max_nights=3,
            flights_data=TWO_CITY_FLIGHTS,
            weather_data=TWO_CITY_WEATHER,
            weather_constraints=WeatherConstraints(),
        )
        result2, _ = solve_itinerary(inp)
        self.assertIsNone(result2)

    def test_selects_cheapest_city_subset(self):
        # 3 candidates, pick 2 — cheapest pair includes Barcelona ($170)
        result, _ = solve_itinerary(_three_city_inp())
        self.assertIsNotNone(result)
        self.assertIn("Barcelona", result.city_order)
        self.assertAlmostEqual(result.total_cost_usd, 170.0, places=2)

    def test_weather_constraint_excludes_city(self):
        # Barcelona has 95% rain chance; constraint cap is 50% → BCN filtered out entirely
        # Cheapest remaining pair: Madrid → Lisbon = $240
        result, _ = solve_itinerary(_three_city_inp(
            weather=THREE_CITY_WEATHER_BAD_BCN,
            constraints=WeatherConstraints(max_rain_pct=50.0),
        ))
        self.assertIsNotNone(result)
        self.assertNotIn("Barcelona", result.city_order)
        self.assertAlmostEqual(result.total_cost_usd, 240.0, places=2)


# ---------------------------------------------------------------------------
# Tool tests  (mock search_flights and search_weather)
# ---------------------------------------------------------------------------

class TestOptimizeItineraryTool(unittest.TestCase):

    _TOOL_ARGS = {
        "candidate_cities": "Barcelona,Madrid",
        "city_iata_pairs": "Barcelona:BCN,Madrid:MAD",
        "origin_city": "Munich",
        "origin_iata": "MUC",
        "num_to_visit": 2,
        "start_date": "2026-06-01",
        "end_date": "2026-06-07",
        "min_nights": 2,
        "max_nights": 3,
    }

    @patch("tools.search_weather")
    @patch("tools.search_flights")
    def test_returns_cheapest_route_as_string(self, mock_flights, mock_weather):
        mock_flights.invoke.side_effect = _mock_flights
        mock_weather.invoke.side_effect = _mock_weather

        from tools import optimize_itinerary
        result = optimize_itinerary.invoke(self._TOOL_ARGS)

        self.assertIsInstance(result, str)
        self.assertIn("Barcelona", result)
        self.assertIn("Madrid", result)
        self.assertIn("Munich", result)
        self.assertIn("$270.00", result)

    @patch("tools.search_weather")
    @patch("tools.search_flights")
    def test_no_flights_returns_infeasible_message(self, mock_flights, mock_weather):
        mock_flights.invoke.side_effect = lambda args: []
        mock_weather.invoke.side_effect = _mock_weather

        from tools import optimize_itinerary
        result = optimize_itinerary.invoke(self._TOOL_ARGS)

        self.assertIn("no itinerary satisfied all constraints", result)

    @patch("tools.search_weather")
    @patch("tools.search_flights")
    def test_search_flights_is_called(self, mock_flights, mock_weather):
        mock_flights.invoke.side_effect = _mock_flights
        mock_weather.invoke.side_effect = _mock_weather

        from tools import optimize_itinerary
        optimize_itinerary.invoke(self._TOOL_ARGS)

        self.assertTrue(mock_flights.invoke.called)

    @patch("tools.search_weather")
    @patch("tools.search_flights")
    def test_search_weather_is_called(self, mock_flights, mock_weather):
        mock_flights.invoke.side_effect = _mock_flights
        mock_weather.invoke.side_effect = _mock_weather

        from tools import optimize_itinerary
        optimize_itinerary.invoke(self._TOOL_ARGS)

        self.assertTrue(mock_weather.invoke.called)

    @patch("tools.search_weather")
    @patch("tools.search_flights")
    def test_result_contains_flight_details(self, mock_flights, mock_weather):
        mock_flights.invoke.side_effect = _mock_flights
        mock_weather.invoke.side_effect = _mock_weather

        from tools import optimize_itinerary
        result = optimize_itinerary.invoke(self._TOOL_ARGS)

        self.assertIn("LH1000", result)   # Lufthansa flight number on cheapest outbound
        self.assertIn("$100.00", result)  # outbound price


if __name__ == "__main__":
    unittest.main(verbosity=2)
