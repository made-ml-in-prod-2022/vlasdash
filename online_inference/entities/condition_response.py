from pydantic import BaseModel


class ConditionResponse(BaseModel):
    """Response from app."""

    id: int
    condition: int
