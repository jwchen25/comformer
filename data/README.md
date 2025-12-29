# Materials Project Data Download

This folder contains scripts to download and process crystal structure data from the Materials Project.

## File Description

### `download_mp_data.py`

A Python script that downloads crystal structures with band gap data from the Materials Project and saves them to an ASE extxyz format file.

**Features:**
- Downloads structures from Materials Project using the official API
- Filters for structures with calculated band gap values
- Converts pymatgen structures to ASE Atoms objects
- Saves all properties (band gap, direct/indirect, formula, material ID) to extxyz file
- Provides logging and statistics

**Requirements:**
- `pymatgen>=2024.1.1`
- `mp-api>=0.35.0` (Materials Project official API)
- `ase>=3.23.0`
- Materials Project API key

**Installation:**

The required dependencies are already listed in `pyproject.toml`. Install with:

```bash
pip install -e ../
```

**Usage:**

1. **Get a Materials Project API key:**
   - Sign up at https://materialsproject.org
   - Get your API key from your account settings

2. **Set the API key as an environment variable:**
   ```bash
   export MP_API_KEY="your_api_key_here"
   ```

3. **Run the script:**
   ```bash
   python download_mp_data.py
   ```

4. **Optional arguments:**
   ```bash
   # Specify output file
   python download_mp_data.py --output my_structures.extxyz --target band_gap

   # Pass API key directly
   python download_mp_data.py --api-key "your_key" --output data.extxyz
   ```

**Output:**

The script generates an extxyz file with the following structure:
- One frame per crystal structure
- Atomic positions and cell parameters
- Additional properties stored as key-value pairs:
  - `mpid`: Materials Project ID
  - `formula`: Chemical formula
  - `TARGET VALUE`: target material property
