from emit.ccl.schedule.schedule_tree import *
from emit.ccl.schedule.schedule_visitor import ScheduleVisitor
from emit.tree import *

if __name__ == '__main__':
    v = ScheduleVisitor()

    s = ScheduleRoot(
        storage_section=StorageSection([
            StorageLine([
                StorageDecl(
                    Identifier('evol_group'),
                    Integer(2)
                )
            ])
        ]),
        schedule_section=ScheduleSection([
            ScheduleBlock(
                group_or_function=GroupOrFunction.Function,
                name=Identifier('HeatEqn_Initialize'),
                at_or_in=AtOrIn.At,
                schedule_bin=Identifier('CCTK_INITIAL'),
                description=String('Initialize evolved variables'),
                lang=Language.C,
                writes=[
                    Intent(Identifier('U'), IntentRegion.Everywhere)
                ],
                reads=[
                    Intent(Identifier('Grid::coordinates'), IntentRegion.Everywhere)
                ]
            ),
            ScheduleBlock(
                group_or_function=GroupOrFunction.Function,
                name=Identifier('HeatEqn_Update'),
                at_or_in=AtOrIn.At,
                schedule_bin=Identifier('CCTK_EVOL'),
                description=String('Evolve the heat equation'),
                lang=Language.C,
                writes=[
                    Intent(Identifier('U'), IntentRegion.Interior)
                ],
                reads=[
                    Intent(Identifier('U_p'), IntentRegion.Everywhere)
                ]
            ),
            ScheduleBlock(
                group_or_function=GroupOrFunction.Function,
                name=Identifier('HeatEqn_Boundary'),
                at_or_in=AtOrIn.At,
                schedule_bin=Identifier('CCTK_EVOL'),
                description=String('Heat equation BC'),
                after=[Identifier('HeatEqn_Update')],
                lang=Language.C,
                writes=[
                    Intent(Identifier('U'), IntentRegion.Boundary)
                ],
                sync=[Identifier('evol_group')]
            )
        ])
    )

    print(v.visit(s))
