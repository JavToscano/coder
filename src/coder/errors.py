from __future__ import annotations


class CoderError(Exception):
    """Base class for errors that should be shown to the user without a traceback."""


class ConfigError(CoderError):
    pass


class CredentialsError(CoderError):
    pass


class ProviderError(CoderError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class OfferUnavailable(ProviderError):
    """The chosen offer was rented by someone else or withdrawn."""


class NoOffersError(CoderError):
    pass


class ProvisioningError(CoderError):
    pass


class SSHError(CoderError):
    pass


class OpenCodeConfigError(CoderError):
    pass
