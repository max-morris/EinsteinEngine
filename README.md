# Einstein Engine
The Einstein Engine is a DSL and toolkit for creating Cactus thorns.
It can generate complete CarpetX thorns from simple recipes written in Python.

## How to set up EinsteinEngine:

1. Create a venv and activate it. 
   ```bash
   python -m venv venv && . ./venv/bin/activate
   ```
2. Install the dependencies.
   ```bash
   python -m pip install -r requirements.txt
   ```

3. Install EinsteinEngine. 
   ```bash
   python -m pip install .
   ```
   
## How to generate a thorn:

1. Be sure the venv is activated.
   ```bash
   . ./venv/bin/activate
   ```

2. Typecheck the recipe with MyPy.
   ```bash
   mypy ./path/to/recipe.py
   ```

3. Run the recipe.
   ```bash
   python ./path/to/recipe.py
   ```
