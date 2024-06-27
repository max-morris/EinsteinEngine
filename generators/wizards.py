import os

from dsl.use_indices import ThornDef
from emit.ccl.interface.interface_visitor import InterfaceVisitor
from emit.ccl.param.param_visitor import ParamVisitor
from emit.ccl.schedule.schedule_visitor import ScheduleVisitor
from emit.code.cpp.cpp_visitor import CppVisitor
from nrpy.helpers.conditional_file_updater import ConditionalFileUpdater

from generators.cpp_carpetx_generator import CppCarpetXGenerator


def cpp_carpetx_wizard(gf: ThornDef) -> None:
    base_dir = os.path.join(gf.arrangement, gf.name)
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "src"), exist_ok=True)

    carpetx_generator = CppCarpetXGenerator(gf)

    for fn_name in gf.thorn_functions.keys():
        print('=====================')
        code_tree = carpetx_generator.generate_function_code(fn_name)
        code = CppVisitor(carpetx_generator).visit(code_tree)
        # print(code)
        code_fname = os.path.join(base_dir, "src", carpetx_generator.get_src_file_name(fn_name))
        with ConditionalFileUpdater(code_fname) as fd:
            fd.write(code)

    print('== param.ccl ==')
    param_tree = carpetx_generator.generate_param_ccl()
    param_ccl = ParamVisitor().visit(param_tree)
    # print(param_ccl)
    param_ccl_fname = os.path.join(base_dir, "param.ccl")
    with ConditionalFileUpdater(param_ccl_fname) as fd:
        fd.write(param_ccl)

    print('== interface.ccl ==')
    interface_tree = carpetx_generator.generate_interface_ccl()
    interface_ccl = InterfaceVisitor().visit(interface_tree)
    # print(interface_ccl)
    interface_ccl_fname = os.path.join(base_dir, "interface.ccl")
    with ConditionalFileUpdater(interface_ccl_fname) as fd:
        fd.write(interface_ccl)

    print('== schedule.ccl ==')
    schedule_tree = carpetx_generator.generate_schedule_ccl()
    schedule_ccl = ScheduleVisitor().visit(schedule_tree)
    # print(schedule_ccl)
    schedule_ccl_fname = os.path.join(base_dir, "schedule.ccl")
    with ConditionalFileUpdater(schedule_ccl_fname) as fd:
        fd.write(schedule_ccl)

    print('== configuration.ccl ==')
    configuration_ccl = """
# Configuration definitions for thorn WaveToyNRPy
REQUIRES Arith Loop
    """.strip()
    # print(configuration_ccl)
    configuration_ccl_fname = os.path.join(base_dir, "configuration.ccl")
    with ConditionalFileUpdater(configuration_ccl_fname) as fd:
        fd.write(configuration_ccl)

    print('== make.code.defn ==')
    makefile = carpetx_generator.generate_makefile()
    # print(makefile)
    makefile_fname = os.path.join(base_dir, "src/make.code.defn")
    with ConditionalFileUpdater(makefile_fname) as fd:
        fd.write(makefile)
