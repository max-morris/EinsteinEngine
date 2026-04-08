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

4. (Optional; recommended for development) Install repository git hooks for this clone.
   ```bash
   bash scripts/install-git-hooks.sh
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

# License
Copyright © 2024--2026 Max Morris, Steven R. Brandt, and other Einstein Engine contributors.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
