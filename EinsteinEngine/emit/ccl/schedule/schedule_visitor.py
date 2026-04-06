#  Copyright (C) 2024-2026 Max Morris, Steven R. Brandt, and other Einstein Engine contributors.
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

from typing import Any

from multimethod import multimethod

from EinsteinEngine.emit.ccl.schedule.schedule_tree import ScheduleNode, ScheduleRoot, StorageSection, StorageLine, \
    StorageDecl, ScheduleSection, ScheduleBlock, GroupOrFunction, Intent
from EinsteinEngine.emit.tree import Identifier, Integer, Verbatim, String, Bool, Float
from EinsteinEngine.emit.visitor import Visitor, visit_each
from EinsteinEngine.util import indent


class ScheduleVisitor(Visitor[ScheduleNode]):

    @multimethod
    def visit(self, n: ScheduleNode) -> Any:
        self.not_implemented(n)

    @visit.register
    def _(self, n: Identifier) -> str:
        return n.identifier

    @visit.register
    def _(self, n: Integer) -> str:
        return f'{n.integer}'

    @visit.register
    def _(self, n: Verbatim) -> str:
        return n.text

    @visit.register
    def _(self, n: String) -> str:
        return f'"{n.text}"' if not n.single_quotes else f"'{n.text}'"

    @visit.register
    def _(self, n: Bool) -> str:
        return "true" if n.b else "false"

    @visit.register
    def _(self, n: Float) -> str:
        return f'{n.fl}'

    @visit.register
    def _(self, n: ScheduleRoot) -> str:
        return '\n'.join(visit_each(self, [n.storage_section, n.schedule_section]))

    @visit.register
    def _(self, n: StorageSection) -> str:
        return '\n'.join(visit_each(self, n.lines))

    @visit.register
    def _(self, n: StorageLine) -> str:
        return f'STORAGE: {", ".join(visit_each(self, n.decls))}'

    @visit.register
    def _(self, n: StorageDecl) -> str:
        return f'{self.visit(n.group)}[{self.visit(n.time_levels)}]'

    @visit.register
    def _(self, n: ScheduleSection) -> str:
        return '\n\n'.join(visit_each(self, n.schedule_blocks))

    @visit.register
    def _(self, n: ScheduleBlock) -> str:
        s: str = 'SCHEDULE '

        if n.group_or_function is GroupOrFunction.Group:
            s += 'GROUP '

        s += f'{self.visit(n.name)} {n.at_or_in.representation} {self.visit(n.schedule_bin)}'

        if n.alias is not None:
            s += f' AS {self.visit(n.alias)}'

        if n.while_var is not None:
            s += f' WHILE {self.visit(n.while_var)}'

        if n.if_var is not None:
            s += f' IN {self.visit(n.if_var)}'

        if n.before is not None and (before_len := len(n.before)) > 0:
            if before_len == 1:
                s += f' BEFORE {self.visit(n.before[0])}'
            else:
                s += f' BEFORE ({" ".join(visit_each(self, n.before))})'

        if n.after is not None and (after_len := len(n.after)) > 0:
            if after_len == 1:
                s += f' AFTER {self.visit(n.after[0])}'
            else:
                s += f' AFTER ({" ".join(visit_each(self, n.after))})'

        s += '\n{'

        s_inner = ''

        if n.lang is not None:
            s_inner += f'\nLANG: {n.lang.representation}'

        if n.storage is not None:
            s_inner += f'\n{self.visit(n.storage)}'

        if n.trigger is not None and len(n.trigger) > 0:
            s_inner += f'\nTRIGGER: {", ".join(visit_each(self, n.trigger))}'

        if n.sync is not None and len(n.sync) > 0:
            s_inner += f'\nSYNC: {", ".join(visit_each(self, n.sync))}'

        if n.options is not None and len(n.options) > 0:
            s_inner += f'\nOPTIONS: {", ".join(visit_each(self, n.options))}'

        if n.reads is not None and len(n.reads) > 0:
            s_inner += f'\nReads: {", ".join(visit_each(self, sorted(n.reads, key=repr)))}'

        if n.writes is not None and len(n.writes) > 0:
            s_inner += f'\nWrites: {", ".join(visit_each(self, sorted(n.writes, key=repr)))}'

        s += indent(s_inner) + '\n} ' + self.visit(n.description)

        return s

    @visit.register
    def _(self, n: Intent) -> str:
        if n.region is None:
            return f'{self.visit(n.name)}'
        return f'{self.visit(n.name)}({n.region.representation})'
