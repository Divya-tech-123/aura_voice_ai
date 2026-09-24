"""Tools subsystem for AURA.

Provides executable tools for calculations, weather queries, web searches,
and safe sandboxed file operations.
"""

from app.tools.base import BaseTool
from app.tools.calculator import CalculatorTool, calculate
from app.tools.weather import (
    WeatherTool,
    get_weather,
    BaseWeatherProvider,
    OpenWeatherProvider,
    MockWeatherProvider,
)
from app.tools.search import (
    SearchTool,
    search_web,
    BaseSearchProvider,
    SerpAPISearchProvider,
    MockSearchProvider,
)
from app.tools.files import FileTool, SafeFileManager
from app.tools.registry import ToolRegistry, create_default_registry

__all__ = [
    "BaseTool",
    "CalculatorTool",
    "calculate",
    "WeatherTool",
    "get_weather",
    "BaseWeatherProvider",
    "OpenWeatherProvider",
    "MockWeatherProvider",
    "SearchTool",
    "search_web",
    "BaseSearchProvider",
    "SerpAPISearchProvider",
    "MockSearchProvider",
    "FileTool",
    "SafeFileManager",
    "ToolRegistry",
    "create_default_registry",
]
