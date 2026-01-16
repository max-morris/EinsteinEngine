# This Makefile generates Cottonmouth's thorns

source_folder = recipes/Cottonmouth
target_folder = Cottonmouth

all: $(target_folder)/CottonmouthBSSNOK/interface.ccl \
	 $(target_folder)/CottonmouthDiagLinearWaveID/interface.ccl \
	 $(target_folder)/CottonmouthKerrSchildID/interface.ccl \
	 $(target_folder)/CottonmouthLinearWaveID/interface.ccl \
	 $(target_folder)/CottonmouthTestBSSNOK/interface.ccl

$(target_folder)/CottonmouthBSSNOK/interface.ccl: $(source_folder)/bssnok_unsplit_ricci.py
	python $<

$(target_folder)/CottonmouthDiagLinearWaveID/interface.ccl: $(source_folder)/diagonal_linear_wave.py
	python $<

$(target_folder)/CottonmouthKerrSchildID/interface.ccl: $(source_folder)/kerr_schild.py
	python $<

$(target_folder)/CottonmouthLinearWaveID/interface.ccl: $(source_folder)/linear_wave.py
	python $<

$(target_folder)/CottonmouthTestBSSNOK/interface.ccl: $(source_folder)/test_bssnok.py
	python $<
