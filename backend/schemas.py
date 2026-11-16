from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    weight: float = Field(gt=0)