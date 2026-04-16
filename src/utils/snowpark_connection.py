"""Snowpark session management for connecting to Snowflake."""

import os
from typing import ClassVar

from dotenv import load_dotenv
from snowflake.snowpark import Session


class SnowPark:
    """Wrapper around a Snowflake Snowpark session, configured from environment variables."""

    _shared_session: ClassVar[Session | None] = None

    @staticmethod
    def _connection_params() -> dict[str, str | None]:
        load_dotenv()
        return {
            "user": os.getenv("SNOWFLAKE_USER"),
            "password": os.getenv("SNOWFLAKE_PASSWORD"),
            "account": os.getenv("SNOWFLAKE_ACCOUNT"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        }

    @classmethod
    def get_shared_session(cls, force_reconnect: bool = False) -> Session:
        """Return a process-wide shared Snowpark session."""
        if force_reconnect:
            cls.close_shared_session()
        if cls._shared_session is None:
            cls._shared_session = Session.builder.configs(cls._connection_params()).create()
        return cls._shared_session

    @classmethod
    def close_shared_session(cls) -> None:
        """Close and clear the process-wide shared Snowpark session."""
        if cls._shared_session is None:
            return
        close = getattr(cls._shared_session, "close", None)
        if callable(close):
            close()
        cls._shared_session = None

    def __init__(self) -> None:
        """Create a Snowpark session using SNOWFLAKE_* env vars loaded from .env."""
        self.session = self.get_shared_session()
