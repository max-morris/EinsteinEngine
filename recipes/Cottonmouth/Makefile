# This Makefile generates Cottonmouth's thorns

source_folder = recipes/Cottonmouth
target_folder = generated/Cottonmouth

all: $(target_folder)/CottonmouthBSSNOK/interface.ccl \
	 $(target_folder)/CottonmouthGaugeWaveID/interface.ccl \
	 $(target_folder)/CottonmouthLinearWaveID/interface.ccl \
	 $(target_folder)/CottonmouthZ4c/interface.ccl \

$(target_folder):
	mkdir -p $(target_folder)

$(target_folder)/CottonmouthBSSNOK/interface.ccl: $(source_folder)/bssnok.py | $(target_folder)
	python $<

$(target_folder)/CottonmouthGaugeWaveID/interface.ccl: $(source_folder)/gauge_wave.py | $(target_folder)
	python $<

$(target_folder)/CottonmouthLinearWaveID/interface.ccl: $(source_folder)/linear_wave.py | $(target_folder)
	python $<

$(target_folder)/CottonmouthZ4c/interface.ccl: $(source_folder)/Z4c.py | $(target_folder)
	python $<
