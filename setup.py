#  Copyright (C) 2024-2026 Max Morris and other Einstein Engine contributors.
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
    python_requires='>=3.13',
    description='DSL and toolset for creating Cactus thorns',
    url='https://github.com/max-morris/EinsteinEngine',
    author='Max Morris',
    author_email='mmorris@cct.lsu.edu',
    license='AGPL-3.0-or-later',
    packages=find_packages(include=['EinsteinEngine', 'EinsteinEngine.*']),
    package_data={'EinsteinEngine': ['py.typed']},
    install_requires=[
        'mypy==2.3.0',
        'nrpy==2.0.18',
        'sympy==1.14.0',
        'multimethod>=1.10',
        'numpy>=2.1.0',
        'scipy>=1.17.1',
        'pdoc>=14.6.0',
        'sortedcontainers==2.4.0',
        'sortedcontainers-stubs==2.4.3',
        'optuna==4.9.0',
        'bayesian-optimization==3.3.0',
        'termcolor==3.3.0'
    ]
)
