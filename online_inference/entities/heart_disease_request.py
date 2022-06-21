from pydantic import BaseModel, validator
from fastapi import HTTPException

AGE_LOWER_BOUND = 0
AGE_UPPER_BOUND = 130
MALE = 1
FEMALE = 0
CHEST_TYPES = [0, 1, 2, 3]
TRESTBPS_LOWER_BOUND = 30
TRESTBPS_UPPER_BOUND = 230
CHOL_LOWER_BOUND = 100
CHOL_UPPER_BOUND = 600
RESTECG_TYPES = [0, 1, 2]
FBS_HIGH_TRUE = 1
FBS_HIGH_FALSE = 0
EXANG_YES = 1
EXANG_NO = 0
THALACH_LOWER_BOUND = 60
THALACH_UPPER_BOUND = 250
OLDPEAK_LOWER_BOUND = 0
OLDPEAK_UPPER_BOUND = 7
SLOPE_TYPES = [0, 1, 2]
CA_TYPES = [0, 1, 2, 3]
THAL_TYPES = [0, 1, 2]


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
        if value < AGE_LOWER_BOUND or value > AGE_UPPER_BOUND:
            raise HTTPException(
                status_code=400,
                detail="Unacceptable age value"
            )

        return value

    @validator('sex')
    def check_sex(cls, value) -> int:
        if value != MALE and value != FEMALE:
            raise HTTPException(
                status_code=400,
                detail="Sex must be 0 or 1"
            )

        return value

    @validator('cp')
    def check_cp(cls, value) -> int:
        if value not in CHEST_TYPES:
            raise HTTPException(
                status_code=400,
                detail="Chest pain type must be 0, 1, 2 or 3"
            )

        return value

    @validator('trestbps')
    def check_trestbps(cls, value) -> int:
        if value < TRESTBPS_LOWER_BOUND or value > TRESTBPS_UPPER_BOUND:
            raise HTTPException(
                status_code=400,
                detail="Unacceptable blood pressure value"
            )

        return value

    @validator('chol')
    def check_chol(cls, value) -> int:
        if value < CHOL_LOWER_BOUND or value > CHOL_UPPER_BOUND:
            raise HTTPException(
                status_code=400,
                detail="Unacceptable serum cholestoral value"
            )

        return value

    @validator('restecg')
    def check_restecg(cls, value) -> int:
        if value not in RESTECG_TYPES:
            raise HTTPException(
                status_code=400,
                detail="Resting electrocardiographic results must be 0, 1 or 2"
            )

        return value

    @validator('fbs')
    def check_fbs(cls, value) -> int:
        if value != FBS_HIGH_TRUE and value != FBS_HIGH_FALSE:
            raise HTTPException(
                status_code=400,
                detail="Fasting blood sugar must be 0 or 1"
            )

        return value

    @validator('exang')
    def check_exang(cls, value) -> int:
        if value != EXANG_NO and value != EXANG_YES:
            raise HTTPException(
                status_code=400,
                detail="Exercise induced angina must be 0 or 1"
            )

        return value

    @validator('thalach')
    def check_thalach(cls, value) -> int:
        if value < THALACH_LOWER_BOUND or value > THALACH_UPPER_BOUND:
            raise HTTPException(
                status_code=400,
                detail="Unacceptable maximum heart rate value"
            )

        return value

    @validator('oldpeak')
    def check_oldpeak(cls, value) -> float:
        if value < OLDPEAK_LOWER_BOUND or value > OLDPEAK_UPPER_BOUND:
            raise HTTPException(
                status_code=400,
                detail="Unacceptable ST depression induced value"
            )

        return value

    @validator('slope')
    def check_slope(cls, value) -> int:
        if value not in SLOPE_TYPES:
            raise HTTPException(
                status_code=400,
                detail="Unacceptable slope value"
            )

        return value

    @validator('ca')
    def check_ca(cls, value) -> int:
        if value not in CA_TYPES:
            raise HTTPException(
                status_code=400,
                detail="Unacceptable number of major vessels"
            )

        return value

    @validator('thal')
    def check_thal(cls, value) -> int:
        if value not in THAL_TYPES:
            raise HTTPException(
                status_code=400,
                detail="Unacceptable thal value"
            )

        return value
