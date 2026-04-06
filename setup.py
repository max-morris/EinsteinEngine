#  Copyright (C) 2024-2026 Max Morris, Steven R. Brandt, and other Einstein Engine contributors.
#
#  This file is part of the Einstein Engine (EinsteinEngine).
#
#  EinsteinEngine is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  EinsteinEngine is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

from setuptools import setup, find_packages

setup(
    name='EinsteinEngine',
    version='0.1.0',
    description='DSL and toolset for creating Cactus thorns',
    url='https://github.com/mmor115/EinsteinEngine',
    author='Max Morris',
    author_email='mmorris@cct.lsu.edu',
    license='MIT',
    packages=find_packages(include='EinsteinEngine.*'),
    install_requires=[
        'mypy==1.16.1',
        'nrpy==2.0.18',
        'sympy==1.12.1',
        'multimethod>=1.10',
        'numpy==2.1.0',
        'pdoc==14.6.0',
        'sortedcontainers==2.4.0',
        'sortedcontainers-stubs==2.4.3'
    ]
)
