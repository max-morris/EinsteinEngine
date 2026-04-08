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

from EinsteinEngine.emit.ccl.interface.interface_tree import InterfaceRoot, HeaderSection, IncludeSection, FunctionSection, \
    VariableSection, VariableGroup, Access, DataType, GroupType
from EinsteinEngine.emit.ccl.interface.interface_visitor import InterfaceVisitor
from EinsteinEngine.emit.tree import Identifier, Integer, String

if __name__ == '__main__':
    v = InterfaceVisitor()

    i = InterfaceRoot(
        HeaderSection(
            implements=Identifier('HeatEqn'),
            inherits=[Identifier('Grid')],
            friends=[]
        ),
        IncludeSection([]),
        FunctionSection([]),
        VariableSection([
            VariableGroup(
                access=Access.Public,
                group_name=Identifier('evol_group'),
                data_type=DataType.Real,
                group_type=GroupType.GF,
                time_levels=Integer(2),
                variable_names=[Identifier('U')],
                group_description=String("Heat equation fields")
            )
        ])
    )

    print(v.visit(i))
