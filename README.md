# Travel Advisor Chatbot

## Setup

Create a `.env` file in the project root with the following keys:

```
OPENROUTER_API_KEY=your_openrouter_api_key
SERPAPI_KEY=your_serpapi_key
WEATHER_API_KEY=your_weatherapi_key
```

| Variable | Where to get it |
|----------|----------------|
| `OPENROUTER_API_KEY` | [openrouter.ai/keys](https://openrouter.ai/keys) — used for the LLM and text embeddings |
| `SERPAPI_KEY` | [serpapi.com/manage-api-key](https://serpapi.com/manage-api-key) — used for flight and hotel search |
| `WEATHER_API_KEY` | [weatherapi.com](https://www.weatherapi.com/my/) — used for weather forecasts |

## Running

### Backend

```bash
pip install -r requirements.txt
python api.py
```

The API will be available at `http://localhost:8001`.

### Frontend

```bash
cd frontend
npm install   # first time only
npm run dev
```

The frontend will be available at `http://localhost:5173`.


### Sprint 2

The Sprint 2 is in the sprint2 branch.
