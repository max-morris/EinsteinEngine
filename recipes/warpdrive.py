from EmitCactus import *
from sympy import Expr

tdf = ThornDef("WD","WD")

vb = tdf.add_param(
    "vb",
    default=1.0,
    desc="Bubble speed"
)

vd = tdf.add_param(
    "vd",
    default=1.0,
    desc="Bubble drag"
)

t, x, y, z = tdf.mk_coords(with_time=True)
one = sympify(1)
zero:Expr = sympify(0)

E = tdf.decl("E",[])
alp = tdf.decl("alp",[])

fx = tdf.decl_fun("vx", 3)
fy = tdf.decl_fun("vy", 3)
fz = tdf.decl_fun("vz", 3)
ffx = -fx(x,y,z)
ffy = -fy(x,y,z)
ffz = -fz(x,y,z)

beta = tdf.decl("beta",[ua], substitution_rule=[ffx,ffy,ffz])
velx = mkSymbol("velx")
vely = mkSymbol("vely")
velz = mkSymbol("velz")
v = tdf.decl("v",[ua],substitution_rule=[velx,vely,velz])

flat_metric = mkMatrix([
[1,0,0],
[0,1,0],
[0,0,1]])

g = tdf.decl("g",[la,lb],symmetries=[(la,lb)],substitution_rule=flat_metric)
tdf.add_substitution_rule(g[ua,ub], flat_metric)

k = tdf.decl("k",[la,lb],symmetries=[(la,lb)], substitution_rule=(1/(2*alp))*(D(beta[uc]*g[la,lc],lb) + D(beta[uc]*g[lb,lc],la)))

# Eq 25
print(tdf.expand(E*(alp*k[lj,lk]*v[uj]*v[uk] - v[uj]*D(alp,lj))))
# Eq 28 a
dqdt = tdf.decl("dqdt",[ui],substitution_rule=alp*v[ui] - beta[ui])
# Eq 28 b
dvdt = tdf.decl("dvdt",[ui],substitution_rule=alp*v[uj]*( v[ui]*(D(log(alp), lj)-k[lj,lk]*v[uk]) + 2*k[lk,lj]*g[uk,ui] ) - g[ui,uj]*D(alp,lj) - v[uj]*D(beta[ui],lj))
