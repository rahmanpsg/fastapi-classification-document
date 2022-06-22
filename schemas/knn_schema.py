from pydantic import BaseModel


class KNNBase(BaseModel):
    text: str
    k: int
