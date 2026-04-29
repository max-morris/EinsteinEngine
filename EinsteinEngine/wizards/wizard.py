#  Copyright (C) 2026 Max Morris and other Einstein Engine contributors.
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

from __future__ import annotations

from abc import ABC
from typing import Generic, TypeVar, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import SupportsWrite

F = TypeVar('F')  # todo: generic Frontend bound
G = TypeVar('G')  # todo: generic Generator bound
CV = TypeVar('CV')  # todo: generic Visitor bound

class Wizard(ABC, Generic[F, G, CV]):
    frontend: F
    generator: G
    code_visitor: CV
    license_header: Optional[str]

    def __init__(self, frontend: F, generator: G, code_visitor: CV, *,
                 license_header: Optional[str] = None):
        self.frontend = frontend
        self.generator = generator
        self.code_visitor = code_visitor
        self.license_header = license_header

    @staticmethod
    def _commentify(s: str, char: str) -> str:
        lines = s.splitlines(keepends=True)
        if not lines:
            lines = [""]

        commented_lines: list[str] = []
        for line in lines:
            newline = ""
            content = line
            if line.endswith("\r\n"):
                content = line[:-2]
                newline = "\r\n"
            elif line.endswith("\n") or line.endswith("\r"):
                content = line[:-1]
                newline = line[-1]

            commented_lines.append(f"{char} {content}{newline}")

        return "".join(commented_lines)

    def write_with_header(self, fd: SupportsWrite[str], s: str, comment_char: Optional[str] = None) -> object:
        if self.license_header is not None:
            fd.write(self._commentify(self.license_header, comment_char) if comment_char is not None else self.license_header)
            fd.write("\n\n")
        return fd.write(s)
