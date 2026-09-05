"""Climbing the inference ladder on one mechanistic model."""

from inferladder.data import Dataset, default_sir_dataset, default_sir_model, make_synthetic
from inferladder.models.base import GenerativeModel, ObservationModel, SystemModel
from inferladder.models.sir import SIR
from inferladder.observation import NegativeBinomialReporting, PoissonReporting

__version__ = "0.2.0"

__all__ = [
    "SIR",
    "Dataset",
    "GenerativeModel",
    "NegativeBinomialReporting",
    "ObservationModel",
    "PoissonReporting",
    "SystemModel",
    "default_sir_dataset",
    "default_sir_model",
    "make_synthetic",
]
