import os
import openai
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type


class RateLimiter:

    @staticmethod
    def with_retry(fn):
        return retry(
            retry=retry_if_exception_type(openai.RateLimitError),
            wait=wait_exponential(multiplier=1, min=4, max=60),
            stop=stop_after_attempt(5),
            reraise=True
        )(fn)

    @staticmethod
    def get_rate_limit_headers() -> dict:
        client = openai.OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        with client.with_streaming_response.chat.completions.create(
            model="openai/gpt-5.4-mini",
            messages=[{"role": "user", "content": "ping"}]
        ) as resp:
            for _ in resp.iter_lines():
                pass
            return {k: v for k, v in resp.headers.items() if "ratelimit" in k.lower()}
