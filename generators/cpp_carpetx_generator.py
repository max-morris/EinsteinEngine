import typing

from dsl.use_indices import ThornDef, ThornFunction
from emit.code.code_tree import *
from emit.tree import String, Identifier
from generators.generator_exception import GeneratorException


class CppCarpetXGenerator:
    thorn_def: ThornDef

    boilerplate_includes: list[Identifier] = [Identifier(s) for s in
                                              ["fixmath.hxx", "cctk.h", "cctk_Arguments.h", "cctk_Parameters.h",
                                               "loop_device.hxx", "simd.hxx", "cmath", "tuple"]]

    boilerplate_namespace_usings: list[Identifier] = [Identifier(s) for s in ["Arith", "Loop"]]

    boilerplate_usings: list[Identifier] = [Identifier(s) for s in ["std::cbrt", "std::fmax", "std::fmin", "std::sqrt"]]

    def __init__(self, thorn_def: ThornDef) -> None:
        self.thorn_def = thorn_def

    def generate_function_code(self, which_fn: str) -> CodeRoot:
        nodes: list[CodeElem] = list()
        thorn_fn: ThornFunction = self.thorn_def.thorn_functions[which_fn]
        fn_name: str = thorn_fn.name

        for include in self.boilerplate_includes:
            nodes.append(IncludeDirective(include))

        for ns in self.boilerplate_namespace_usings:
            nodes.append(UsingNamespace(ns))

        nodes.append(Using(self.boilerplate_usings))

        bases_of_outputs = {self.thorn_def.base_of[base] for base in
                            {str(output) for output in thorn_fn.eqnlist.outputs} if base in self.thorn_def.base_of}
        output_centerings = {self.thorn_def.centering[base] for base in bases_of_outputs}

        if None in output_centerings or len(output_centerings) == 0:
            raise GeneratorException("All output vars must have a centering.")

        if len(output_centerings) > 1:
            raise GeneratorException(f"Output vars have mixed centerings: {output_centerings}")

        output_centering: Centering
        [output_centering] = typing.cast(set[Centering], output_centerings)

        nodes.append(
            ThornFunctionDecl(
                Identifier(fn_name),
                [DeclareCarpetXArgs(Identifier(fn_name)),
                 DeclareCarpetParams(),
                 CarpetXGridLoopCall(
                     output_centering,
                     CarpetXGridLoopLambda(
                         {str(lhs): SympyExpr(rhs) for lhs, rhs in thorn_fn.eqnlist.eqns.items()})
                 )]
            )
        )

        return CodeRoot(nodes)
