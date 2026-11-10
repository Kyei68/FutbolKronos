"""Núcleo de red de FutbolKronos, derivado de Kronos (MIT, ver LICENSE_Kronos.txt).

Se llama kronos_futbol (no `model`, como en Kronos-code) para que no choque con el paquete
`model` del Kronos financiero que usa la carpeta superior en el mismo entorno de Python.
"""
from .kronos import Kronos, KronosTokenizer
from .futbol import FutbolKronos, cargar, logits_marcador

model_dict = {
    "kronos_tokenizer": KronosTokenizer,
    "kronos": Kronos,
    "futbol_kronos": FutbolKronos,
}


def get_model_class(model_name):
    if model_name in model_dict:
        return model_dict[model_name]
    raise NotImplementedError(f"Modelo {model_name} no encontrado en model_dict")
