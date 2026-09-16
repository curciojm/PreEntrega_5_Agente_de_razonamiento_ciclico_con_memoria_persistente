from enum import Enum

from pydantic import BaseModel, Field


class ResultadoConcepto(BaseModel):
    fuente: str = Field(description="Fuente bibliográfica del fragmento recuperado.")
    contenido: str = Field(description="Contenido relevante recuperado del documento.")


class ResultadoFuente(BaseModel):
    fuente: str = Field(description="Fuente bibliográfica del documento.")
    # Se fuerza a entero porque Pinecone puede devolver el número de página como float.
    pagina: int = Field(description="Número entero de página donde aparece la información.")

class LLMErrorType(str, Enum):
    """Tipos de errores utilizados para clasificar las excepciones."""

    RATE_LIMIT = "rate_limit"
    UNKNOWN = "unknown"
    KEY = "key"


class LLMError(Exception):
    """Permite clasificar el error y proporcionar un mensaje legible al usuario."""

    def __init__(self, error_type: str, message: str):
        self.error_type = error_type
        self.message = message
        super().__init__(message)