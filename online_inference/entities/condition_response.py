from pydantic import BaseModel


class ConditionResponse(BaseModel):
    id: int
    condition: int
