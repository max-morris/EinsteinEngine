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

from EinsteinEngine.emit.ccl.schedule.schedule_tree import ScheduleRoot, StorageSection, StorageLine, StorageDecl, \
    ScheduleSection, ScheduleBlock, GroupOrFunction, AtOrIn, Intent, IntentRegion
from EinsteinEngine.emit.ccl.schedule.schedule_visitor import ScheduleVisitor
from EinsteinEngine.emit.tree import Identifier, Integer, String, Language

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
