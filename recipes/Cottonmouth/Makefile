# This Makefile generates Cottonmouth's thorns

source_folder = recipes/Cottonmouth
target_folder = generated/Cottonmouth
generated_root = generated

all: $(target_folder)/CottonmouthBSSNOK4m/interface.ccl \
	 $(target_folder)/CottonmouthGaugeWaveID/interface.ccl \
	 $(target_folder)/CottonmouthLinearWaveID/interface.ccl \
	 $(target_folder)/CottonmouthZ4c4m/interface.ccl \
	 $(target_folder)/CottonmouthBosonStarID/interface.ccl \
	 $(target_folder)/CottonmouthBS4m/interface.ccl \

$(target_folder):
	mkdir -p $(target_folder)

$(target_folder)/CottonmouthBSSNOK4m/interface.ccl: $(source_folder)/bssnok.py | $(target_folder)
	cd $(generated_root) && python ../$<

$(target_folder)/CottonmouthGaugeWaveID/interface.ccl: $(source_folder)/gauge_wave.py | $(target_folder)
	cd $(generated_root) && python ../$<

$(target_folder)/CottonmouthLinearWaveID/interface.ccl: $(source_folder)/linear_wave.py | $(target_folder)
	cd $(generated_root) && python ../$<

$(target_folder)/CottonmouthZ4c4m/interface.ccl: $(source_folder)/Z4c.py | $(target_folder)
	cd $(generated_root) && python ../$<

$(target_folder)/CottonmouthBosonStarID/interface.ccl: $(source_folder)/boson_star_id.py $(source_folder)/boson_star_profile.py $(source_folder)/bs_profile_table.hxx | $(target_folder)
	cd $(generated_root) && python ../$<

$(target_folder)/CottonmouthBS4m/interface.ccl: $(source_folder)/boson_star.py $(source_folder)/boson_star_profile.py $(source_folder)/bs_profile_table.hxx | $(target_folder)
	cd $(generated_root) && python ../$< --metric-thorn CottonmouthZ4c4m --nstars 1
