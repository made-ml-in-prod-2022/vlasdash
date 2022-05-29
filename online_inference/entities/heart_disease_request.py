from pydantic import BaseModel, validator
from fastapi import HTTPException


class HeartDiseaseRequest(BaseModel):
    """Request from app."""

    id: int
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

    @validator('id')
    def check_id(cls, value) -> int:
        if value < 0:
            raise HTTPException(
                status_code=400,
                detail="Id must be non-negative"
            )

        return value

    @validator('age')
    def check_age(cls, value) -> int:
        if value <= 0:
            raise HTTPException(
                status_code=400,
                detail="Age must be greater than 0"
            )
        if value > 130:
            raise HTTPException(
                status_code=400,
                detail="Age must be less than 130"
            )

        return value

    @validator('sex')
    def check_sex(cls, value) -> int:
        if value != 0 and value != 1:
            raise HTTPException(
                status_code=400,
                detail="Sex must be 0 or 1"
            )

        return value

    @validator('cp')
    def check_cp(cls, value) -> int:
        if value not in [0, 1, 2, 3]:
            raise HTTPException(
                status_code=400,
                detail="Chest pain type must be 0, 1, 2 or 3"
            )

        return value

    @validator('trestbps')
    def check_trestbps(cls, value) -> int:
        if value > 230 or value < 30:
            raise HTTPException(
                status_code=400,
                detail="Unacceptable blood pressure value"
            )

        return value

    @validator('chol')
    def check_chol(cls, value) -> int:
        if value > 600 or value < 100:
            raise HTTPException(
                status_code=400,
                detail="Unacceptable serum cholestoral value"
            )

        return value

    @validator('restecg')
    def check_restecg(cls, value) -> int:
        if value not in [0, 1, 2]:
            raise HTTPException(
                status_code=400,
                detail="Resting electrocardiographic results must be 0, 1 or 2"
            )

        return value

    @validator('fbs')
    def check_fbs(cls, value) -> int:
        if value != 0 and value != 1:
            raise HTTPException(
                status_code=400,
                detail="Fasting blood sugar must be 0 or 1"
            )

        return value

    @validator('exang')
    def check_exang(cls, value) -> int:
        if value != 0 and value != 1:
            raise HTTPException(
                status_code=400,
                detail="Exercise induced angina must be 0 or 1"
            )

        return value

    @validator('thalach')
    def check_thalach(cls, value) -> int:
        if value < 60 or value > 250:
            raise HTTPException(
                status_code=400,
                detail="Unacceptable maximum heart rate value"
            )

        return value

    @validator('oldpeak')
    def check_oldpeak(cls, value) -> float:
        if value < 0 or value > 7:
            raise HTTPException(
                status_code=400,
                detail="Unacceptable ST depression induced value"
            )

        return value

    @validator('slope')
    def check_slope(cls, value) -> int:
        if value not in [0, 1, 2]:
            raise HTTPException(
                status_code=400,
                detail="Unacceptable slope value"
            )

        return value

    @validator('ca')
    def check_ca(cls, value) -> int:
        if value not in [0, 1, 2, 3]:
            raise HTTPException(
                status_code=400,
                detail="Unacceptable number of major vessels"
            )

        return value

    @validator('thal')
    def check_thal(cls, value) -> int:
        if value not in [0, 1, 2]:
            raise HTTPException(
                status_code=400,
                detail="Unacceptable thal value"
            )

        return value
