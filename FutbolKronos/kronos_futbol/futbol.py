"""
FutbolKronos: el transformer de Kronos con cabezas propias para el marcador.

- cabeza="softmax": como siempre, 25 probabilidades libres (una por marcador 0-4+ × 0-4+).
- cabeza="poisson": el contexto de [PARTIDO] da los goles esperados de cada equipo (λ local,
  λ visitante) y un parámetro de dependencia ρ (Dixon-Coles); de ahí salen los 25 marcadores
  ("4+" = cola de la Poisson). Menos parámetros y marcadores coherentes entre sí.
- n_rasgos > 0 (híbrido): una red pequeña sobre los rasgos numéricos de futbol_rasgos.py suma
  su aportación a los 25 logits (softmax) o a log λ (poisson). Se normalizan con la media y la
  escala del entrenamiento, que se guardan con el modelo.

Las demás posiciones de la secuencia se siguen entrenando con la cabeza de tokens de Kronos
(pérdida auxiliar). Los modelos antiguos (Kronos a secas) se siguen cargando como antes.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

from .kronos import Kronos

N_SCORES, MAX_GOALS = 25, 4


class FutbolKronos(nn.Module, PyTorchModelHubMixin):
    def __init__(self, kronos: dict, cabeza: str = "softmax", n_rasgos: int = 0,
                 rasgos_media: list = None, rasgos_escala: list = None, dropout_rasgos: float = 0.3):
        super().__init__()
        assert cabeza in ("softmax", "poisson")
        self.cabeza, self.n_rasgos = cabeza, int(n_rasgos)
        self.m = Kronos(**kronos)
        d = kronos["d_model"]
        if cabeza == "poisson":
            self.lam = nn.Linear(d, 2)
            nn.init.zeros_(self.lam.weight)
            self.lam.bias.data = torch.tensor([math.log(1.45), math.log(1.10)])
            self.rho = nn.Parameter(torch.tensor(-0.05))
        if self.n_rasgos:
            out = N_SCORES if cabeza == "softmax" else 2
            self.rasgos = nn.Sequential(nn.Linear(self.n_rasgos, 64), nn.GELU(), nn.Dropout(dropout_rasgos),
                                        nn.Linear(64, 64), nn.GELU(), nn.Dropout(dropout_rasgos), nn.Linear(64, out))
            nn.init.zeros_(self.rasgos[-1].weight)
            nn.init.zeros_(self.rasgos[-1].bias)
            media = rasgos_media if rasgos_media is not None else [0.0] * self.n_rasgos
            escala = rasgos_escala if rasgos_escala is not None else [1.0] * self.n_rasgos
            self.register_buffer("r_media", torch.tensor(media, dtype=torch.float32))
            self.register_buffer("r_escala", torch.tensor(escala, dtype=torch.float32))

    def _rasgos(self, R):
        x = (R.float() - self.r_media) / self.r_escala
        return self.rasgos(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))

    def _poisson(self, loglam):
        loglam = loglam.clamp(-3.0, 2.5)
        k = torch.arange(MAX_GOALS, device=loglam.device, dtype=loglam.dtype)
        lf = torch.lgamma(k + 1)

        def logp(x):                                       # log P(0..3) y log P(4+)
            base = (k[None] * x[:, None] - x.exp()[:, None] - lf[None]).exp()
            return torch.cat([base, (1 - base.sum(1, keepdim=True)).clamp_min(1e-6)], 1).log()

        J = logp(loglam[:, 0])[:, :, None] + logp(loglam[:, 1])[:, None, :]
        lh, la, r = loglam[:, 0].exp(), loglam[:, 1].exp(), self.rho
        adj = torch.zeros_like(J)
        adj[:, 0, 0] = torch.log((1 - lh * la * r).clamp_min(1e-3))
        adj[:, 0, 1] = torch.log((1 + lh * r).clamp_min(1e-3))
        adj[:, 1, 0] = torch.log((1 + la * r).clamp_min(1e-3))
        adj[:, 1, 1] = torch.log((1 - r).clamp_min(1e-3))
        return (J + adj).reshape(len(loglam), N_SCORES)

    def marcador(self, ctx, logits_ultimo, R=None):
        """Logits de los 25 marcadores a partir del contexto de la posición [PARTIDO]."""
        if self.cabeza == "poisson":
            loglam = self.lam(ctx)
            if self.n_rasgos:
                loglam = loglam + self._rasgos(R)
            return self._poisson(loglam)
        out = logits_ultimo[:, :N_SCORES]
        if self.n_rasgos:
            out = out + self._rasgos(R)
        return out

    def logits_marcador(self, s1, s2, stamp, R=None):
        """Entrada terminada en [PARTIDO] (sin [MARCADOR]) -> logits (N, 25)."""
        l1, ctx = self.m.decode_s1(s1, s2, stamp)
        return self.marcador(ctx[:, -1], l1[:, -1], R)


def logits_marcador(model, s1, s2, stamp, R=None):
    """Igual para FutbolKronos y para un Kronos antiguo (cabeza de tokens)."""
    if isinstance(model, FutbolKronos):
        return model.logits_marcador(s1, s2, stamp, R)
    return model.decode_s1(s1, s2, stamp)[0][:, -1, :N_SCORES]


def cargar(ruta):
    """Carga un modelo guardado, sea FutbolKronos o un Kronos antiguo."""
    import json, os
    with open(os.path.join(ruta, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    return (FutbolKronos if "kronos" in cfg else Kronos).from_pretrained(ruta)
