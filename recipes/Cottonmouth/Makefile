# This Makefile generates Cottonmouth's thorns

source_folder = recipes/Cottonmouth
target_folder = Cottonmouth

all: $(target_folder)/CottonmouthBSSNOK/interface.ccl \
	 $(target_folder)/CottonmouthDiagLinearWaveID/interface.ccl \
	 $(target_folder)/CottonmouthLinearWaveID/interface.ccl \
	 $(target_folder)/CottonmouthZ4c/interface.ccl \

$(target_folder)/CottonmouthBSSNOK/interface.ccl: $(source_folder)/bssnok_unsplit_ricci_opt.py
	python $<

$(target_folder)/CottonmouthDiagLinearWaveID/interface.ccl: $(source_folder)/diagonal_linear_wave.py
	python $<

$(target_folder)/CottonmouthLinearWaveID/interface.ccl: $(source_folder)/linear_wave.py
	python $<

$(target_folder)/CottonmouthZ4c/interface.ccl: $(source_folder)/Z4c.py
	python $<
