from EmitCactus import *

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

E = tdf.decl("E",[])
alp = tdf.decl("alp",[])

f = tdf.decl_fun("f", 1)
ff = vd*f((x-vb*t)**2 + y**2 + z**2)

beta = tdf.decl("beta",[ua], substitution_rule=[ff,0,0])
v = tdf.decl("v",[ua])

flat_metric = mkMatrix([
[1,0,0],
[0,1,0],
[0,0,1]])

g = tdf.decl("g",[la,lb],symmetries=[(la,lb)],substitution_rule=flat_metric)
tdf.add_substitution_rule(g[ua,ub], flat_metric)

k = tdf.decl("k",[la,lb],symmetries=[(la,lb)], substitution_rule=(1/(2*alp))*(D(beta[uc]*g[la,lc],lb) + D(beta[uc]*g[lb,lc],la)))

dEdt = E*(alp*k[lj,lk]*v[uj]*v[uk] - v[uj]*D(alp,lj))
dqdt = tdf.decl("dqdt",[ui],substitution_rule=alp*v[ui] - beta[ui])
dvdt = tdf.decl("dvdt",[ui],substitution_rule=alp*v[uj]*( v[ui]*(D(log(alp), lj)-k[lj,lk]*v[uk]) + 2*k[lk,lj]*g[uk,ui] ) - g[ui,uj]*D(alp,lj) - v[uj]*D(beta[ui],lj))
