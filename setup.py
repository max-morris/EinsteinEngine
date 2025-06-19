from setuptools import setup, find_packages

setup(
    name='EmitCactus',
    version='0.1.0',
    description='DSL and toolset for creating Cactus thorns',
    url='https://github.com/mmor115/EmitCactus',
    author='Max Morris',
    author_email='mmorris@cct.lsu.edu',
    license='MIT',
    packages=find_packages(include='EmitCactus.*'),
    install_requires=[
        'mypy==1.16.1',
        'nrpy==2.0.18',
        'sympy==1.12.1',
        'multimethod>=1.10',
        'numpy==2.1.0',
        'pdoc==14.6.0'
    ]
)
