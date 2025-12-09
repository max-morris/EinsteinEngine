from EmitCactus import *

thorn = ThornDef('Test', 'Abcd')

opts = {
    "do_split_output_eqns": False
}

a = thorn.add_param('a', 1.0, 'a')
b = thorn.add_param('b', 1.0, 'b')
c = thorn.add_param('c', 1.0, 'c')

o1 = thorn.decl('o1', [])
o2 = thorn.decl('o2', [])

f1 = thorn.create_function('f1', ScheduleBin.Evolve)
f2 = thorn.create_function('f2', ScheduleBin.Evolve)

f1.add_eqn(o1, a + b + c)
f2.add_eqn(o2, a + b)

f1.bake(**opts)
f2.bake(**opts)

print(">"*50)

thorn.do_global_cse()

CppCarpetXWizard(
    thorn,
    CppCarpetXGenerator(
        thorn
    )
).generate_thorn()
