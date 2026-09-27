"""Registry and discovery for Country and Road-Standard Profiles."""
from typing import Dict, Any, List, Optional
from backend.app.intelligence.profiles.base import BaseCountryProfile
from backend.app.intelligence.profiles.usa import USAProfile
from backend.app.intelligence.profiles.india import IndiaProfile


class ProfileRegistry:
    """Registry managing available national road-standard and regulatory profiles."""

    def __init__(self):
        self._profiles: Dict[str, BaseCountryProfile] = {}
        # Pre-register built-in profiles
        self.register(USAProfile())
        self.register(IndiaProfile())

    def register(self, profile: BaseCountryProfile) -> None:
        """Registers a country profile instance by its lowercase code."""
        self._profiles[profile.code.lower().strip()] = profile

    def get(self, code: Optional[str] = None) -> BaseCountryProfile:
        """Retrieves profile by country code.

        Defaults to 'usa' if code is not provided or not found.
        """
        if not code:
            return self._profiles["usa"]
        normalized = code.lower().strip()
        return self._profiles.get(normalized, self._profiles["usa"])

    def list_profiles(self) -> List[Dict[str, Any]]:
        """Returns summary metadata for all registered profiles."""
        return [p.get_summary() for p in self._profiles.values()]

    def is_supported(self, code: str) -> bool:
        """Checks if a country code is registered."""
        return code.lower().strip() in self._profiles


# Global profile registry singleton
profile_registry = ProfileRegistry()

__all__ = [
    "BaseCountryProfile",
    "USAProfile",
    "IndiaProfile",
    "ProfileRegistry",
    "profile_registry",
]
