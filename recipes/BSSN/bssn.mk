all : PyBSSN/BSSN/.make PyBSSN/KerrSchildID/.make PyBSSN/LinearWaveID/.make PyBSSN/DiagonalLinearWaveID/.make PyBSSN/TestBSSN/.make
PyBSSN/BSSN/.make : recipes/BSSN/bssn.py
	python3 ./recipes/BSSN/bssn.py && touch PyBSSN/BSSN/.make
PyBSSN/KerrSchildID/.make : recipes/BSSN/kerr_schild.py
	python3 ./recipes/BSSN/kerr_schild.py && touch PyBSSN/KerrSchildID/.make
PyBSSN/LinearWaveID/.make : recipes/BSSN/linear_wave.py
	python3 ./recipes/BSSN/linear_wave.py && touch PyBSSN/LinearWaveID/.make
PyBSSN/DiagonalLinearWaveID/.make : recipes/BSSN/diagonal_linear_wave.py
	python3 ./recipes/BSSN/diagonal_linear_wave.py && touch PyBSSN/DiagonalLinearWaveID/.make
PyBSSN/TestBSSN/.make : recipes/BSSN/test/kerr_schild.par
	python3 ./recipes/BSSN/test_bssn.py && touch PyBSSN/TestBSSN/.make
