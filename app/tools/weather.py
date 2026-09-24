"""Weather lookup tool for AURA.

Fetches current weather and atmospheric conditions for a requested city.
Decouples the tool execution interface from underlying external API providers
with robust configuration checks and error handling.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import requests

from app.tools.base import BaseTool

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Weather Provider Interface & Implementations
# ==============================================================================

class BaseWeatherProvider(ABC):
    """Abstract interface for weather data providers."""

    @abstractmethod
    def get_weather(self, city: str) -> Dict[str, Any]:
        """Fetch weather data for the specified city.

        Args:
            city: Name of the city to look up.

        Returns:
            Dict with 'success' (bool), 'city' (str), 'data' (dict/None), and 'error' (str/None).
        """
        pass


class OpenWeatherProvider(BaseWeatherProvider):
    """Production weather provider utilizing OpenWeatherMap API."""

    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 10.0) -> None:
        self.api_key = api_key or os.getenv("WEATHER_API_KEY", "").strip() or os.getenv("OPENWEATHER_API_KEY", "").strip()
        self.timeout = timeout

    def get_weather(self, city: str) -> Dict[str, Any]:
        clean_city = (city or "").strip()
        if not clean_city:
            return {
                "success": False,
                "city": "",
                "data": None,
                "error": "City name cannot be empty.",
            }

        # Clear configuration error when credentials are not configured
        if not self.api_key:
            return {
                "success": False,
                "city": clean_city,
                "data": None,
                "error": (
                    "Weather API key is not configured. "
                    "Please set WEATHER_API_KEY in your .env file."
                ),
            }

        params = {
            "q": clean_city,
            "appid": self.api_key,
            "units": "metric",
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=self.timeout)
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "city": clean_city,
                "data": None,
                "error": f"Weather API request timed out after {self.timeout}s.",
            }
        except requests.exceptions.RequestException as exc:
            return {
                "success": False,
                "city": clean_city,
                "data": None,
                "error": f"Failed to connect to weather provider: {exc}",
            }

        if response.status_code == 401:
            return {
                "success": False,
                "city": clean_city,
                "data": None,
                "error": "Authentication failed: Invalid weather API key (HTTP 401).",
            }
        elif response.status_code == 404:
            return {
                "success": False,
                "city": clean_city,
                "data": None,
                "error": f"City '{clean_city}' was not found.",
            }
        elif response.status_code == 429:
            return {
                "success": False,
                "city": clean_city,
                "data": None,
                "error": "Weather API rate limit reached (HTTP 429).",
            }
        elif response.status_code >= 400:
            return {
                "success": False,
                "city": clean_city,
                "data": None,
                "error": f"Weather provider returned error status {response.status_code}.",
            }

        try:
            payload = response.json()
            main = payload.get("main", {})
            weather_desc = payload.get("weather", [{}])[0].get("description", "Unknown").title()
            wind = payload.get("wind", {})

            return {
                "success": True,
                "city": payload.get("name", clean_city),
                "data": {
                    "temperature": main.get("temp"),
                    "feels_like": main.get("feels_like"),
                    "humidity": main.get("humidity"),
                    "condition": weather_desc,
                    "wind_speed": wind.get("speed"),
                    "units": "celsius",
                },
                "error": None,
            }
        except Exception as exc:
            return {
                "success": False,
                "city": clean_city,
                "data": None,
                "error": f"Failed to parse weather API response: {exc}",
            }


class MockWeatherProvider(BaseWeatherProvider):
    """Offline, deterministic weather provider for testing and offline development."""

    def __init__(
        self,
        canned_weather: Optional[Dict[str, Dict[str, Any]]] = None,
        simulated_error: Optional[str] = None,
    ) -> None:
        self.canned_weather = canned_weather or {
            "london": {
                "temperature": 16.5,
                "feels_like": 15.8,
                "humidity": 78,
                "condition": "Light Rain",
                "wind_speed": 4.1,
                "units": "celsius",
            },
            "tokyo": {
                "temperature": 22.0,
                "feels_like": 21.5,
                "humidity": 60,
                "condition": "Clear Sky",
                "wind_speed": 3.2,
                "units": "celsius",
            },
            "new york": {
                "temperature": 19.2,
                "feels_like": 18.9,
                "humidity": 55,
                "condition": "Partly Cloudy",
                "wind_speed": 5.0,
                "units": "celsius",
            },
        }
        self.simulated_error = simulated_error

    def get_weather(self, city: str) -> Dict[str, Any]:
        if self.simulated_error:
            return {
                "success": False,
                "city": city,
                "data": None,
                "error": self.simulated_error,
            }

        clean = (city or "").strip()
        if not clean:
            return {
                "success": False,
                "city": "",
                "data": None,
                "error": "City name cannot be empty.",
            }

        city_key = clean.lower()
        if city_key in self.canned_weather:
            return {
                "success": True,
                "city": clean.title(),
                "data": self.canned_weather[city_key],
                "error": None,
            }

        return {
            "success": True,
            "city": clean.title(),
            "data": {
                "temperature": 20.0,
                "feels_like": 19.5,
                "humidity": 50,
                "condition": "Clear",
                "wind_speed": 3.0,
                "units": "celsius",
            },
            "error": None,
        }


def get_weather_provider(provider_name: Optional[str] = None, **kwargs: Any) -> BaseWeatherProvider:
    """Factory to instantiate the appropriate weather provider."""
    name = (provider_name or os.getenv("WEATHER_PROVIDER", "")).strip().lower()
    if name == "mock":
        return MockWeatherProvider(**kwargs)
    return OpenWeatherProvider(**kwargs)


# ==============================================================================
# 2. Tool Implementation & Top-Level Helper
# ==============================================================================

def get_weather(
    city: str,
    api_key: Optional[str] = None,
    provider: Optional[BaseWeatherProvider] = None,
) -> Dict[str, Any]:
    """Fetch current weather data for a city.

    Args:
        city: Name of the city to look up.
        api_key: Optional API key overriding environment configuration.
        provider: Optional custom BaseWeatherProvider instance.

    Returns:
        Dict with 'success' (bool), 'city' (str), 'data' (dict/None), and 'error' (str/None).
    """
    if provider:
        active_provider = provider
    elif api_key:
        active_provider = OpenWeatherProvider(api_key=api_key)
    else:
        active_provider = get_weather_provider()

    return active_provider.get_weather(city=city)


class WeatherTool(BaseTool):
    """AURA executable tool for retrieving current weather reports."""

    name: str = "weather"
    description: str = "Fetches current weather, temperature, and atmospheric conditions for a specified city."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "Name of the city or municipality (e.g. 'London', 'Paris', 'Tokyo').",
            }
        },
        "required": ["city"],
    }

    def __init__(self, provider: Optional[BaseWeatherProvider] = None) -> None:
        self.provider = provider

    def execute(self, city: str = "", **kwargs: Any) -> Dict[str, Any]:
        """Execute the weather tool."""
        return get_weather(city=city, provider=self.provider)
