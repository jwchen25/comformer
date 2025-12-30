import time
from comformer import load_predictor_hf
from pymatgen.core import Structure, Lattice

# Load trained model with specific hf repo and subfolder
predictor = load_predictor_hf(
    repo_id="jwchen25/MatFlow",
    subfolder="property_prediction/mp_bulk_modulus",
)

st = time.time()

a = 5.64
lattice = Lattice.cubic(a)
species = ["Na", "Cl"]
coords = [
    [0, 0, 0],        # Na
    [0.5, 0.5, 0.5]   # Cl
]
nacl = Structure(lattice, species, coords)
strucs = [nacl for i in range(200)]
print(predictor.predict(strucs, batch_size=200))

et = time.time()
print(f"total time: {et-st} seconds")
