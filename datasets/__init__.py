from .datasets_banknote import load_banknote
from .datasets_heart import load_heart
from .datasets_car import load_car
from .datasets_wine import load_wine
from .datasets_moons import load_moons
from .datasets_circles import load_circles

def get_all_datasets():
    return {
        "Banknote": load_banknote(),
        "Heart": load_heart(),
        "Car": load_car(),
        "Wine": load_wine(),
        "Moons": load_moons(),
        "Circles": load_circles(),
    }
