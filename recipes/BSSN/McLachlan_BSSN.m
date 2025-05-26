(* Boilerplate setup *)

SetEnhancedTimes[False];



(******************************************************************************)
(* Various Mathematica definitions that may depend on Kranc, but which
   are independent of McLachlan *)
(******************************************************************************)

(* Define a tensor and declare its symmetries *)
(* Note: We don't want to use (or write down) an expression such as
   g[la,lb] before g has been declared as tensor, since g[la,lb] is
   then apparently not interpreted correctly. Therefore, tensor name
   and index list are passed separately. *)
(* Note: syms is a list of symmetries, where each symmetry is a list
   of two indices. *)
(* Example call: DefineTensor1[g, {la,lb}, {{la,lb}}] *)
DefineTensor1[name_, indices_:{}, syms_:{}] :=
  Module[{tensor},
         DefineTensor[name];
         tensor = If[indices=={}, name, name[Sequence@@indices]];
         AssertSymmetricIncreasing[tensor, Sequence@@#]& /@ syms;
         tensor];

(* Define a group and declare its name *)
DefineGroup1[name_, tensor_, timelevels_:1] :=
  Module[{group},
         group = CreateGroupFromTensor[tensor];
         group = AddGroupExtra[group, Timelevels -> timelevels];
         group = SetGroupName[group, name];
         group];



(* Split a calculation *)
(* calc:    calculation
   suffix:  suffix to be added to the original name
   updates: keyword-value pairs that should be added to or replaced in
            the calculation
   vars:    left hand sides of the equations that should be kept
            (together with their dependencies) *)
PartialCalculation[calc_, suffix_, updates_, vars_] :=
  Module[
    {name, calc1, replacements, calc2, vars1, patterns, eqs, calc3},
    (* Add suffix to name *)
    name  = lookupDefault[calc, Name, ""] <> suffix;
    calc1 = mapReplaceAdd[calc, Name, name];
    (* Replace some entries in the calculation *)
    replacements = updates //. (lhs_ -> rhs_) -> (mapReplaceAdd[#, lhs, rhs]&);
    calc2        = calc1 // Composition@@replacements;
    (* Remove unnecessary equations *)
    vars1    = Join[vars, lookupDefault[calc2, Shorthands, {}]];
    patterns = Replace[vars1, {Tensor[n_,__]      ->     Tensor[n,__] ,
                               dot[Tensor[n_,__]] -> dot[Tensor[n,__]]}, 1];
    eqs      = FilterRules[lookup[calc, Equations], patterns];
    calc3    = mapReplace[calc2, Equations, eqs];
    calc3];



(******************************************************************************)
(* Choose code to generate *)
(******************************************************************************)

prefix = "ML_";
name = Environment["ML_CODE"];
thorn = prefix <> name;

(* default settings *)
useOpenCL  = False;
useVectors = True;

evolveAKranc = evolveA;
evolveBKranc = evolveB;

addMatterKranc = 1;

addDissipationKranc = 1;

maxTimelevels = 4 (*3*);

(* TODO: splitUpwindDerivsKranc==False doesn't work, leads to "dir1
   not declared" errors *)
derivOrder             = 4;
splitUpwindDerivsKranc = True;



(* NOTE: Variables named "xxxKranc" ared used by Kranc at build time.
   They can either be set to a particular value, and Kranc will then
   generate specialized code. They can also be set to "xxx" where
   "xxx" needs to be a parameter declared below; in this case, Kranc
   will generate generic code that makes this choice at run time. *)
Switch[name,

       (* Standard: a generic BSSN implementation *)
       "BSSN",
       formulationKranc = fBSSN;
       conformalMethodKranc = conformalMethod,

       (* Benchmark: A highly optimized version of BSSN *)
       "BSSN_bench",
       formulationKranc = fBSSN;
       conformalMethodKranc = conformalMethod;
       evolveAKranc = 0;
       evolveBKranc = 1,

       (* Benchmark: Another highly optimized version of BSSN *)
       "BSSN_bench8",
       formulationKranc = fBSSN;
       conformalMethodKranc = cmW;
       evolveAKranc = 0;
       evolveBKranc = 0;
       addDissipationKranc = 0;
       derivOrder = 8,

       (* OpenCL: OpenCL code is optimized at run time depending on
          parameter settings; we therefore choose very generic
          parameter settings *)
       "BSSN_CL",
       formulationKranc = formulation;
       conformalMethodKranc = conformalMethod;
       useOpenCL = True,

       (* No dissipation *)
       "BSSN_ND",
       formulationKranc = fBSSN;
       conformalMethodKranc = conformalMethod;
       addDissipationKranc = 0,

       (* No Vectorization: This is mostly for debugging. Test as much
          as possible, by choosing generic parameter settings *)
       "BSSN_NV",
       formulationKranc = formulation;
       conformalMethodKranc = conformalMethod;
       useVectors = False,

       "CCZ4",
       formulationKranc = fCCZ4;
       (* TODO: is this correct? see e.g. Ricci tensor *)
       conformalMethodKranc = cmW];



(******************************************************************************)
(* Global choices (e.g. for formulations) *)
(******************************************************************************)

(* Global choices are evaluated either at Kranc time, or at run time
   via a Cactus parameter. They cannot differ for different grid
   points. Mathematica variables with the suffix "Kranc" make
   Kranc-time choices. These variables are either set to a small
   integer, determining the choice at Kranc time, or are set to a
   Cactus parameter name, delaying the choice until run time. *)

(* In the If expressions below, the Unevaluated empty Sequence is even
   "less empty" than Null (the empty argument). This is convenient for
   eating a preceding comma: The expression
      { phi->0, IfCCZ4[Theta->0] }
   becomes either
      { phi->0, Theta->0 }
   or
      { phi->0 }
   (note that there is no trailing comma).
   *)

(* Choices that can be either Kranc time or run time *)

(* Formulation (BSSN or CCZ4) *)
fBSSN=0; fCCZ4=1;
IfCCZ4[expr_] :=
  Which[formulationKranc === fBSSN, Unevaluated[Sequence[]],
        True, expr];
IfCCZ4[expr_, else_] :=
  Which[formulationKranc === fCCZ4, expr,
        formulationKranc === fBSSN, else,
        True, IfThen[formulationKranc != fBSSN, expr, else]];

(* Conformal treatment (phi or W) *)
(* This could be made a run-time choice *)
cmPhi=0; cmW=1;
IfW[expr_, else_] :=
  Which[conformalMethodKranc === cmW, expr,
        conformalMethodKranc === cmPhi, else,
        True, IfThen[conformalMethodKranc != cmPhi, expr, else]];

(* Include matter terms? *)
IfMatter[expr_] :=
  Which[addMatterKranc === 0, Unevaluated[Sequence[]],
        True, expr];
IfMatter[expr_, else_] :=
  Which[addMatterKranc === 1, expr,
        addMatterKranc === 0, else,
        True, IfThen[addMatterKranc != 0, expr, else]];

(* Is A (time derivative of lapse) evolved? *)
IfA[expr_] :=
  Which[evolveAKranc === 0, Unevaluated[Sequence[]],
        True, expr];
IfA[expr_, else_] :=
  Which[evolveAKranc === 1, expr,
        evolveAKranc === 0, else,
        True, IfThen[evolveAKranc != 0, expr, else]];

(* Is B^i (time derivative of shift) evolved? *)
IfB[expr_] :=
  Which[evolveBKranc === 0, Unevaluated[Sequence[]],
        True, expr];
IfB[expr_, else_] :=
  Which[evolveBKranc === 1, expr,
        evolveBKranc === 0, else,
        True, IfThen[evolveBKranc != 0, expr, else]];

(* Is dissipation added? *)
IfDiss[expr_, else_] :=
  Which[addDissipationKranc === 0, else,
        True, expr];

(* Run-time choices *)

(* Shift formulation *)
shiftGammaDriver=0; shiftHarmonic=1;
(*
ShiftChoice[exprGammaDriver_, exprHarmonic_] :=
  IfThen[shiftFormulation == shiftGammaDriver, exprGammaDriver, exprHarmonic];
*)



(******************************************************************************)
(* Declare tensors *)
(******************************************************************************)

(* Rename some Cactus grid functions *)

(* CartGrid3D coordinates *)
x1=x; x2=y; x3=z;

(* ADMBase variables *)
admg11=gxx; admg12=gxy; admg22=gyy; admg13=gxz; admg23=gyz; admg33=gzz;
admK11=kxx; admK12=kxy; admK22=kyy; admK13=kxz; admK23=kyz; admK33=kzz;
admalpha=alp;
admdtalpha=dtalp;
admbeta1=betax; admbeta2=betay; admbeta3=betaz;
admdtbeta1=dtbetax; admdtbeta2=dtbetay; admdtbeta3=dtbetaz;

(* TmunuBase variables *)
T00=eTtt;
T01=eTtx; T02=eTty; T03=eTtz;
T11=eTxx; T12=eTxy; T22=eTyy; T13=eTxz; T23=eTyz; T33=eTzz;



(* Tensors *)

DefineTensor1[dir, {ua}];
DefineTensor1[epsdiss, {ua}];

(* BSSN/CCZ4 state vector *)
DefineTensor1[phi];
DefineTensor1[gt, {la,lb}, {{la,lb}}];
DefineTensor1[Xt, {ua}];
DefineTensor1[trK];
DefineTensor1[At, {la,lb}, {{la,lb}}];
DefineTensor1[Theta];

DefineTensor1[alpha];
DefineTensor1[A];
DefineTensor1[beta, {ua}];
DefineTensor1[B, {ua}];

(* conformal Christoffel and curvature variables *)
DefineTensor1[em4phi];
DefineTensor1[gtu, {ua,ub}, {{ua,ub}}];
DefineTensor1[Gtl, {la,lb,lc}, {{lb,lc}}];
DefineTensor1[Gtlu, {la,lb,uc}];
DefineTensor1[Gt, {ua,lb,lc}, {{lb,lc}}];
DefineTensor1[Xtn, {ua}];
DefineTensor1[Zl, {la}];
DefineTensor1[Z, {ua}];
DefineTensor1[cdphi, {la}];
DefineTensor1[cdphi2, {la,lb}];
DefineTensor1[Atm, {ua,lb}];
DefineTensor1[Atu, {ua,ub}, {{ua,ub}}];
DefineTensor1[Rt, {la,lb}, {{la,lb}}];
DefineTensor1[Rphi, {la,lb}, {{la,lb}}];
DefineTensor1[Ats, {la,lb}, {{la,lb}}];
DefineTensor1[trAts];
DefineTensor1[dotXt, {ua}];
DefineTensor1[dotTheta];
DefineTensor1[dottrK];
DefineTensor1[dotalpha];
DefineTensor1[dotbeta, {ua}];
DefineTensor1[ddetgt, {la}];

(* Matter variables *)
DefineTensor1[rho];
DefineTensor1[S, {la}];
DefineTensor1[trS];
DefineTensor1[T00];
DefineTensor1[T0, {la}];
DefineTensor1[T, {la,lb}, {{la,lb}}];

(* constraints from BSSN decomposition *)
DefineTensor1[cS];
DefineTensor1[cXt, {ua}];
DefineTensor1[cA];

(* ADM quantities *)
DefineTensor1[g, {la,lb}, {{la,lb}}];
DefineTensor1[K, {la,lb}, {{la,lb}}];

DefineTensor1[gu, {ua,ub}, {{ua,ub}}];
DefineTensor1[R, {la,lb}, {{la,lb}}];
DefineTensor1[trR];

(* ADM constraints *)
DefineTensor1[H];
DefineTensor1[M, {la}];

(* ADMBase variables *)
DefineTensor1[admg, {la,lb}, {{la,lb}}];
DefineTensor1[admK, {la,lb}, {{la,lb}}];
DefineTensor1[admalpha];
DefineTensor1[admdtalpha];
DefineTensor1[admbeta, {ua}];
DefineTensor1[admdtbeta, {ua}];



(* These weights are probably unused *)
SetTensorAttribute[phi, TensorWeight, +1/6];
SetTensorAttribute[gt,  TensorWeight, -2/3];
SetTensorAttribute[Xt,  TensorWeight, +2/3];
SetTensorAttribute[At,  TensorWeight, -2/3];
SetTensorAttribute[cS,  TensorWeight, +2  ];
SetTensorAttribute[cXt, TensorWeight, +2/3];



DefineConnection[CD, PD, G];
DefineConnection[CDt, PD, Gt];



(* Groups *)

evolvedGroups = {
  DefineGroup1[prefix <> "log_confac", phi      , maxTimelevels],
  DefineGroup1[prefix <> "metric"    , gt[la,lb], maxTimelevels],
  DefineGroup1[prefix <> "Gamma"     , Xt[ua]   , maxTimelevels],
  DefineGroup1[prefix <> "trace_curv", trK      , maxTimelevels],
  DefineGroup1[prefix <> "curv"      , At[la,lb], maxTimelevels],
  DefineGroup1[prefix <> "Theta"     , Theta    , maxTimelevels] // IfCCZ4,
  DefineGroup1[prefix <> "lapse"     , alpha    , maxTimelevels],
  DefineGroup1[prefix <> "dtlapse"   , A        , maxTimelevels] // IfA,
  DefineGroup1[prefix <> "shift"     , beta[ua] , maxTimelevels],
  DefineGroup1[prefix <> "dtshift"   , B[ua]    , maxTimelevels] // IfB};

(* TODO: isn't there a Theta constraint as well? *)
(* TODO: make the evaluated groups use other_timelevels *)
evaluatedGroups = {
  DefineGroup1[prefix <> "Ham"        , H      , maxTimelevels],
  DefineGroup1[prefix <> "mom"        , M[la]  , maxTimelevels],
  DefineGroup1[prefix <> "cons_detg"  , cS     , maxTimelevels],
  DefineGroup1[prefix <> "cons_Gamma" , cXt[ua], maxTimelevels],
  DefineGroup1[prefix <> "cons_traceA", cA     , maxTimelevels]};

declaredGroups = Join[evolvedGroups, evaluatedGroups];
declaredGroupNames = groupName /@ declaredGroups;

extraGroups =
  {{"grid::coordinates", {x, y, z, r}},
   {"ADMBase::metric",  {gxx, gxy, gxz, gyy, gyz, gzz}},
   {"ADMBase::curv",    {kxx, kxy, kxz, kyy, kyz, kzz}},
   {"ADMBase::lapse",   {alp}},
   {"ADMBase::dtlapse", {dtalp}},
   {"ADMBase::shift",   {betax, betay, betaz}},
   {"ADMBase::dtshift", {dtbetax, dtbetay, dtbetaz}},
   {"TmunuBase::stress_energy_scalar", {eTtt}},
   {"TmunuBase::stress_energy_vector", {eTtx, eTty, eTtz}},
   {"TmunuBase::stress_energy_tensor", {eTxx, eTxy, eTxz, eTyy, eTyz, eTzz}}};

groups = Join[declaredGroups, extraGroups];

inheritedImplementations =
  {"ADMBase", IfMatter["TmunuBase"]};



(******************************************************************************)
(* Cactus Parameters *)
(******************************************************************************)

extendedKeywordParameters = {
  {
    Name -> "ADMBase::evolution_method",
    AllowedValues -> {thorn}
  },
  {
    Name -> "ADMBase::lapse_evolution_method",
    AllowedValues -> {thorn}
  },
  {
    Name -> "ADMBase::shift_evolution_method",
    AllowedValues -> {thorn}
  },
  {
    Name -> "ADMBase::dtlapse_evolution_method",
    AllowedValues -> {thorn}
  },
  {
    Name -> "ADMBase::dtshift_evolution_method",
    AllowedValues -> {thorn}
  }};

keywordParameters = {
  {
    Name -> "initial_boundary_condition",
    Visibility -> "restricted", (* for compatibility *)
    Steerable -> Always,        (* for compatibility *)
    Description -> "Boundary condition for initial condition for some of the BSSN variables",
    AllowedValues -> {{Value -> "scalar", Description -> "not recommended; use ML_BSSN_Helper's value 'extrapolate-gammas' instead"}},
    Default -> "scalar"
  },
  {
    Name -> "rhs_boundary_condition",
    Visibility -> "restricted", (* for compatibility *)
    Steerable -> Always,        (* for compatibility *)
    Description -> "Boundary condition for BSSN RHS and some of the ADMBase variables",
    AllowedValues -> {{Value -> "scalar", Description -> "not recommended; use ML_BSSN_Helper's option 'NewRad' instead"}},
    Default -> "scalar"
  },
  {
    Name -> "rhs_evaluation",
    Visibility -> "restricted", (* for compatibility *)
    Steerable -> Always,        (* for compatibility *)
    Description -> "Whether and how the RHS routine should be split to improve performance",
    AllowedValues -> {{Value -> "combined", Description -> "use a single routine (probably slow)"},
                      (*{Value -> "split3", Description -> "split manually into 3 routines"},*)
                      {Value -> "splitBy", Description -> "split into 3 routines via Kranc"}
                      (*{Value -> "separatedDerivatives", Description -> "not yet implemented"}*)},
    Default -> "splitBy"
  },
  {
    Name -> "my_initial_data",
    Visibility -> "restricted", (* for compatibility *)
    Description -> "(OUTDATED)",
    AllowedValues -> {{Value -> "ADMBase", Description -> "from ADMBase"},
                      (* {Value -> "Minkowski", Description -> "Minkowski"}, *)
                      {Value -> "default", Description -> "do nothing"}},
    Default -> "default"
  },
  {
    Name -> "my_initial_boundary_condition",
    Visibility -> "restricted", (* for compatibility *)
    Description -> "(OUTDATED)",
    AllowedValues -> {{Value -> "none", Description -> "none"},
                      {Value -> "default", Description -> "do nothing"}},
    Default -> "default"
  },
  {
    Name -> "my_rhs_boundary_condition",
    Visibility -> "restricted", (* for compatibility *)
    Description -> "(OUTDATED)",
    AllowedValues -> {{Value -> "none", Description -> "none"},
                      {Value -> "static", Description -> "static"},
                      (* {Value -> "radiative", Description -> "radiative"}, *)
                      {Value -> "default", Description -> "do nothing"}},
    Default -> "default"
  },
  {
    Name -> "my_boundary_condition",
    Visibility -> "restricted", (* for compatibility *)
    Description -> "(OUTDATED)",
    AllowedValues -> {{Value -> "none", Description -> "none"},
                      {Value -> "Minkowski", Description -> "Minkowski"},
                      {Value -> "default", Description -> "do nothing"}},
    Default -> "default"
  },
  {
    Name -> "dt_lapse_shift_method",
    Description -> "(OUTDATED) Treatment of ADMBase dtlapse and dtshift",
    AllowedValues -> {{Value -> "correct", Description -> "(unused)"},
                      {Value -> "noLapseShiftAdvection", Description -> "(unused)"},
                      {Value -> "default", Description -> "do nothing"}},
    Default -> "default"
  },
  {
    Name -> "apply_dissipation",
    Visibility -> "restricted", (* for compatibility *)
    Description -> "(OUTDATED) Whether to apply dissipation to the RHSs",
    (* "yes" and "no" keyword values confuse Cactus, and Kranc doesn't
       support boolean parameters *)
    AllowedValues -> {{Value -> "always", Description -> "yes"},
                      {Value -> "never", Description -> "no"},
                      {Value -> "default", Description -> "do nothing"}},
    Default -> "default"
  }};

intParameters = {
  {
    Name -> fdOrder,
    Description -> "Finite differencing order",
    AllowedValues -> {2, 4, 6, 8},
    Default -> derivOrder
  },
  If[!NumericQ[formulationKranc],
     {
       Name -> formulation,
       Description -> "Formulation",
       AllowedValues -> {{Value -> fBSSN, Description -> "BSSN"},
                         {Value -> fCCZ4, Description -> "CCZ4"}},
       Default -> fBSSN
     },
     Unevaluated[Sequence[]]],
  {
    Name -> conformalMethod,
    Description -> "Treatment of conformal factor",
    AllowedValues -> {If[!NumericQ[conformalMethodKranc] ||
                         conformalMethodKranc === cmPhi,
                         {Value -> cmPhi, Description -> "phi method"},
                         Unevaluated[Sequence[]]],
                      If[!NumericQ[conformalMethodKranc] ||
                         conformalMethodKranc === cmW,
                         {Value -> cmW,   Description -> "W method"},
                         Unevaluated[Sequence[]]]},
    Default -> If[!NumericQ[conformalMethodKranc], cmPhi, conformalMethodKranc]
  },
  {
    Name -> evolveA,
    Steerable -> Always,        (* for compatibility *)
    Description -> "Evolve time derivative of lapse A? (former LapseACoeff)",
    AllowedValues -> {If[!NumericQ[evolveAKranc] || evolveAKranc===0,
                         {Value -> 0, Description -> "off"},
                         Unevaluated[Sequence[]]],
                      If[!NumericQ[evolveAKranc] || evolveAKranc===1,
                         {Value -> 1, Description -> "on"},
                         Unevaluated[Sequence[]]]},
    Default -> If[!NumericQ[evolveAKranc], 0, evolveAKranc]
  },
  {
    Name -> evolveB,
    Steerable -> Always,        (* for compatibility *)
    Visibility -> "restricted", (* for compatibility *)
    Description -> "Evolve time derivative of shift B^i? (former ShiftBCoeff)",
    AllowedValues -> {If[!NumericQ[evolveBKranc] || evolveBKranc===0,
                         {Value -> 0, Description -> "off"},
                         Unevaluated[Sequence[]]],
                      If[!NumericQ[evolveBKranc] || evolveBKranc===1,
                         {Value -> 1, Description -> "on"},
                         Unevaluated[Sequence[]]]},
    Default -> If[!NumericQ[evolveBKranc], 1, evolveBKranc]
  },
  If[!NumericQ[addMatterKranc],
     {
       Name -> addMatter,
       Description -> "Add matter terms?",
       AllowedValues -> {{Value -> 0, Description -> "off"},
                         {Value -> 1, Description -> "on"}},
       Default -> 1
     },
     Unevaluated[Sequence[]]],
  {
    Name -> harmonicN,
    Description -> "d/dt alpha = - f alpha^n K  (harmonic: n=2, 1+log: n=1)",
    Default -> 2
  },
  {
    Name -> shiftFormulation,
    Description -> "shift formulation",
    AllowedValues -> {{Value -> shiftGammaDriver, Description -> "Gamma driver"},
                      {Value -> shiftHarmonic, Description -> "harmonic"}},
    Default -> shiftGammaDriver
  },
  {
    Name -> useSpatialBetaDriver,
    Description -> "Enable spatially varying betaDriver",
    AllowedValues -> {{Value -> 0, Description -> "off"},
                      {Value -> 1, Description -> "on"}},
    Default -> 0
  },
  {
    Name -> useSpatialShiftGammaCoeff,
    Description -> "Enable spatially varying shiftGammaCoeff",
    AllowedValues -> {{Value -> 0, Description -> "off"},
                      {Value -> 1, Description -> "on"}},
    Default -> 0
  },
  {
    Name -> advectLapse,
    Visibility -> "restricted", (* for compatibility *)
    Steerable -> Always,        (* for compatibility *)
    Description -> "Advect lapse? (former LapseAdvectionCoeff)",
    AllowedValues -> {{Value -> 0, Description -> "off"},
                      {Value -> 1, Description -> "on"}},
    Default -> 1
  },
  {
    Name -> advectShift,
    Visibility -> "restricted", (* for compatibility *)
    Steerable -> Always,        (* for compatibility *)
    Description -> "Advect shift? (former ShiftAdvectionCoeff)",
    AllowedValues -> {{Value -> 0, Description -> "off"},
                      {Value -> 1, Description -> "on"}},
    Default -> 1
  },
  {
    Name -> fixAdvectionTerms,
    Description -> "Modify driver and advection terms to work better?",
    AllowedValues -> {{Value -> 0, Description -> "off"},
                      {Value -> 1, Description -> "on"}},
    Default -> 0
  }
};

realParameters = {
  IfCCZ4[{
    Name -> GammaShift,
    Description -> "CCZ4 dovariant shift term in Gamma",
    Default -> 0.5
  }],
  IfCCZ4[{
    Name -> dampk1,
    Description -> "CCZ4 damping term 1 for Theta and Z",
    Default -> 0
  }],
  IfCCZ4[{
    Name -> dampk2,
    Description -> "CCZ4 damping term 2 for Theta and Z",
    Default -> 0
  }],
  {
    Name -> harmonicF,
    Description -> "d/dt alpha = - f alpha^n K   (harmonic: f=1, 1+log: f=2)",
    Default -> 1
  },
  {
    Name -> alphaDriver,
    Description -> "d/dt alpha = ... - alphaDriver (alpha - 1)   (use 1/M (?))",
    Default -> 0 (* TODO: change default? *)
  },
  {
    Name -> shiftGammaCoeff,
    Description -> "d/dt beta^i = C Xt^i   (use C=0.75/M)",
    Default -> 0 (* TODO: change default? *)
  },
  {
    Name -> betaDriver,
    Description -> "d/dt beta^i = ... - betaDriver alpha^shiftAlphaPower beta^i   (use 1/M (?))",
    Default -> 0 (* TODO: change default? *)
  },
  {
    Name -> shiftAlphaPower,
    Description -> "d/dt beta^i = ... - betaDriver alpha^shiftAlphaPower beta^i   (use 0 (?))",
    Default -> 0
  },
  {
    Name -> spatialBetaDriverRadius,
    Description -> "Radius at which the betaDriver starts to be reduced",
    AllowedValues -> {{Value -> "(0:*", Description -> "positive"}},
    Default -> 10^12
  },
  {
    Name -> spatialShiftGammaCoeffRadius,
    Description -> "Radius at which shiftGammaCoeff starts to be reduced",
    AllowedValues -> {{Value -> "(0:*", Description -> "positive"}},
    Default -> 10^12
  },
  {
    Name -> minimumLapse,
    Description -> "Enforced minimum of the lapse function",
    AllowedValues -> {{Value -> "0:*", Description -> "non-negative"}},
    Default -> 0
  },
  (* TODO: provide this parameter only when dissipation is enabled, or
     alternatively, allow only the value zero when dissipation is
     disabled *)
  {
    Name -> epsDiss,
    Visibility -> "restricted", (* for compatibility *)
    Steerable -> Always,        (* for compatibility *)
    Description -> "Dissipation strength",
    AllowedValues -> {{Value -> "0:*", Description -> "non-negative"}},
    Default -> 0
  },
  {
     Name -> LapseACoeff,
     Steerable -> Always, (* for compatibility *)
     Visibility -> "restricted", (* for compatibility *)
     Description -> "(OUTDATED) Evolve time derivative of lapse A? (now evolveA)",
     AllowedValues -> {{Value -> 0.0, Description -> "off"},
                       {Value -> 1.0, Description -> "on"},
                       {Value -> -1.0, Description -> "default"}},
     Default -> -1.0
  },
  {
     Name -> ShiftBCoeff,
     Steerable -> Always, (* for compatibility *)
     Visibility -> "restricted", (* for compatibility *)
     Description -> "(OUTDATED) Evolve time derivative of shift B^i? (now evolveB)",
     AllowedValues -> {{Value -> 0.0, Description -> "off"},
                       {Value -> 1.0, Description -> "on"},
                       {Value -> -1.0, Description -> "default"}},
     Default -> -1.0
  },
  {
    Name -> "LapseAdvectionCoeff",
    Visibility -> "restricted", (* for compatibility *)
    Description -> "(OUTDATED) Advect lapse? (now advectLapse)",
    AllowedValues -> {{Value -> 0.0, Description -> "off"},
                      {Value -> 1.0, Description -> "on"},
                      {Value -> -1.0, Description -> "default"}},
    Default -> -1.0
  },
  {
    Name -> "ShiftAdvectionCoeff",
    Visibility -> "restricted", (* for compatibility *)
    Description -> "(OUTDATED) Advect shift? (now advectShift)",
    AllowedValues -> {{Value -> 0.0, Description -> "off"},
                      {Value -> 1.0, Description -> "on"},
                      {Value -> -1.0, Description -> "default"}},
    Default -> -1.0
  }
};



(******************************************************************************)
(* Derivatives *)
(******************************************************************************)

(* Various Mathematica definitions that involve tensors *)

detgExpr = Det[MatrixOfComponents[g[la,lb]]];
detgtExpr = Det[MatrixOfComponents[gt[la,lb]]];

betaDriverFunc[r_] :=
  IfThen[useSpatialBetaDriver!=0,
         spatialBetaDriverRadius / Max[r, spatialBetaDriverRadius] betaDriver,
         betaDriver];
shiftGammaCoeffFunc[r_] :=
  IfThen[useSpatialShiftGammaCoeff!=0,
         Min[Exp[1 - r / spatialShiftGammaCoeffRadius], 1] shiftGammaCoeff,
         shiftGammaCoeff];



(* Derivatives *)

SCDO = StandardCenteredDifferenceOperator;
SUDO = StandardUpwindDifferenceOperator;

(* Note: SUDO is only expanded when m1 and m2 are integer. It is
   therefore important that SUDO1 is only expanded in this case as
   well, since the replacement is otherwise applied too early. *)
SUDO1[p_, m1_Integer, m2_Integer, i_Integer] :=
  Module[{dir},
         (* We need to expand dir[i] ourselves *)
         dir[j_Integer] := Symbol["dir"<>ToString[j]];
         dir[i] SUDO[p, m1, m2, i] /. shift[j_] -> shift[j]^dir[j]];

(* Yes, the symmetric stencil has a minus sign, and vice versa *)
SUDOsymm[p_, m1_Integer, m2_Integer, i_Integer] :=
  1/2 (SUDO[p, m1, m2, i] - SUDO[p, m2, m1, i])
SUDOanti[p_, m1_Integer, m2_Integer, i_Integer] :=
  1/2 (SUDO[p, m1, m2, i] + SUDO[p, m2, m1, i])

derivatives = {
  PDstandardNth[i_]    -> SCDO[1, fdOrder/2, i],
  PDstandardNth[i_,i_] -> SCDO[2, fdOrder/2, i],
  PDstandardNth[i_,j_] -> SCDO[1, fdOrder/2, i] SCDO[1, fdOrder/2, j],

  PDupwindNth[i_]     -> SUDO1[1, fdOrder/2-1, fdOrder/2+1, i],
  PDupwindNthSymm[i_] -> SUDOsymm[1, fdOrder/2-1, fdOrder/2+1, i],
  PDupwindNthAnti[i_] -> SUDOanti[1, fdOrder/2-1, fdOrder/2+1, i],

  (* Note: for stability (and to reduce the stencil radius), we lower
     the order of accuracy here *)
  PDonesided[i_] -> SUDO1[1, 0, fdOrder/2+1, i],

  PDdissipationNth[i_] -> ((-1)^(fdOrder/2)
                           spacing[i]^(fdOrder+1) / 2^(fdOrder+2)
                           SCDO[fdOrder+2, fdOrder/2+1, i])};

PD     = PDstandardNth;
PDu    = PDupwindNth;
PDua   = PDupwindNthAnti;
PDus   = PDupwindNthSymm;
PDo    = PDonesided;
PDdiss = PDdissipationNth;

If[splitUpwindDerivsKranc,
   Upwind[dir_, var_, idx_] := dir PDua[var,idx] + Abs[dir] PDus[var,idx],
   Upwind[dir_, var_, idx_] := dir PDu[var,idx]];

Dissipation[var_] := IfDiss[epsdiss[ux] PDdiss[var, lx], 0];



(******************************************************************************)
(* Master calculations *)
(******************************************************************************)

BSSNFromMinkowskiCalc = {
  Equations -> {
    phi       -> IfW[1, 0],
    gt[la,lb] -> KroneckerDelta[la,lb],
    Xt[ua]    -> 0,
    trK       -> 0,
    At[la,lb] -> 0,
    Theta     -> 0,
    alpha     -> 1,
    A         -> 0,
    beta[ua]  -> 0,
    B[ua]     -> 0}};



BSSNFromADMBaseCalc = {
  Shorthands -> {dir[ua],
                 g[la,lb], detg, gu[ua,ub],
                 em4phi,
                 gtu[ua,ub], Gt[ua,lb,lc],
                 shiftGammaCoeffValue},
  Equations -> {
    dir[ua] -> Sign[admbeta[ua]],

    g[la,lb]     -> admg[la,lb],
    detg         -> detgExpr,
    gu[ua,ub]    -> detgExpr/detg MatrixInverse[g[ua,ub]],

    phi          -> IfW[detg^(-1/6), Log[detg]/12],
    em4phi       -> IfW[phi^2, Exp[-4 phi]],
    gt[la,lb]    -> em4phi g[la,lb],

    gtu[ua,ub]   -> 1/em4phi gu[ua,ub],
    Gt[ua,lb,lc] -> (1/2 gtu[ua,ud]
                     (PD[gt[lb,ld],lc] + PD[gt[lc,ld],lb] - PD[gt[lb,lc],ld])),
    Xt[ua]       -> gtu[ub,uc] Gt[ua,lb,lc],

    trK          -> gu[ua,ub] admK[la,lb],
    At[la,lb]    -> em4phi * (admK[la,lb] - 1/3 g[la,lb] trK),

    Theta        -> 0,

    alpha        -> admalpha,
    (* See RHS *)
    A            -> IfA[1 / (-harmonicF admalpha^harmonicN)
                        (+ admdtalpha
                         - IfThen[advectLapse!=0,
                                  Upwind[admbeta[ua], admalpha, la],
                                  0]),
                        0],

    shiftGammaCoeffValue -> shiftGammaCoeffFunc[r],

    beta[ua]     -> admbeta[ua],
    (* See RHS *)
    (* Note: cannot have (shiftFormulation == shiftharmonic && evolveB) *)
    (* TODO: check this *)
    B[ua]        -> IfB[1 / (shiftGammaCoeffValue admalpha^shiftAlphaPower)
                        (+ admdtbeta[ua]
                         - IfThen[advectShift!=0,
                                  Upwind[admbeta[ub], admbeta[ua], lb],
                                  0]),
                        0]}};



BSSNToBSSNCalc = {
  Shorthands -> {detgt, gtu[ua,ub], Atm[ua,lb], trAt},
  Equations -> {
    detgt     -> 1 (* TODO COMPATIBILITY detgtExpr *),
    gt[la,lb] -> detgt^(-1/3) gt[la,lb],

    detgt      -> 1 (* detgtExpr *),
    gtu[ua,ub] -> detgtExpr/detgt MatrixInverse[gt[ua,ub]],
    Atm[ua,lb] -> gtu[ua,uc] At[lc,lb],
    trAt       -> Atm[ua,la],
    At[la,lb]  -> At[la,lb] - 1/3 gt[la,lb] trAt,

    alpha -> Max[alpha, minimumLapse]}};



EverythingFromBSSNCalc = {
  Shorthands -> {dir[ua], epsdiss[ua],
                 detgt, gtu[ua,ub], Gtl[la,lb,lc], Gtlu[la,lb,uc], Gt[ua,lb,lc],
                 Xtn[ua],
                 e4phi, em4phi, g[la,lb], gu[ua,ub],
                 Zl[la], Z[ua],
                 fac1, cdphi[la], fac2, cdphi2[la,lb],
                 Rt[la,lb], Rphi[la,lb], R[la,lb], trR,
                 Atm[ua,lb], Atu[ua,ub], trAt,
                 rho, S[la], trS,
                 Ats[la,lb], trAts,
                 betaDriverValue, shiftGammaCoeffValue,
                 dotXt[ua], dotTheta, dottrK, dotalpha, dotbeta[ua],
                 ddetgt[la]},
  Equations -> {
    dir[ua] -> Sign[beta[ua]],
    epsdiss[ua] -> epsDiss,

    (* Connection coefficients *)
    detgt          -> 1 (* detgtExpr *),
    (* This leads to simpler code *)
    gtu[ua,ub]     -> detgtExpr/detgt MatrixInverse[gt[ua,ub]],
    Gtl[la,lb,lc]  -> (1/2 (+ PD[gt[lb,la],lc] + PD[gt[lc,la],lb]
                            - PD[gt[lb,lc],la])),
    Gtlu[la,lb,uc] -> gtu[uc,ud] Gtl[la,lb,ld],
    Gt[ua,lb,lc]   -> gtu[ua,ud] Gtl[ld,lb,lc],

    em4phi    -> IfW[phi^2, Exp[-4 phi]],
    e4phi     -> 1 / em4phi,
    g[la,lb]  -> e4phi gt[la,lb],
    (* detg      -> detgExpr, *)
    gu[ua,ub] -> em4phi gtu[ua,ub],



    (* These conformal connection coefficients are calculated from the
       conformal metric, and are used instead of Xt where no
       derivatives of Xt are taken *)
    Xtn[ua] -> gtu[ub,uc] Gt[ua,lb,lc],

    (* Z quantities *)
    (* gr-qc:1106.2254 (2011), eqn. (23) *)
    Zl[la] -> 1/2 (- PD[gt[la,lb],lc] gtu[ub,uc] + gt[la,lc] Xt[uc]),
    Z[ua]  -> gu[ua,ub] Zl[lb],



    (* Curvature *)

    (* PRD 62 044034 (2000), eqn. (18) *)
    (* CCZ4: Adding Z term by changing Xtn to Xt *)
    Rt[la,lb] -> (- 1/2 gtu[uc,ud] PD[gt[la,lb],lc,ld]
                  + 1/2 gt[lc,la] PD[Xt[uc],lb]
                  + 1/2 gt[lc,lb] PD[Xt[uc],la]
                  + 1/2 Xtn[uc] Gtl[la,lb,lc]
                  + 1/2 Xtn[uc] Gtl[lb,la,lc]
                  + (+ Gt[uc,la,ld] Gtlu[lb,lc,ud]
                     + Gt[uc,lb,ld] Gtlu[la,lc,ud]
                     + Gt[uc,la,ld] Gtlu[lc,lb,ud])),

    fac1          -> IfW[-1/(2 phi), 1],
    cdphi[la]     -> fac1 CDt[phi,la],
    fac2          -> IfW[1/(2 phi^2), 0],
    cdphi2[la,lb] -> fac1 CDt[phi,la,lb] + fac2 CDt[phi,la] CDt[phi,lb],

    (* PRD 62 044034 (2000), eqn. (15) *)
    Rphi[la,lb] -> (- 2 cdphi2[lb,la]
                    - 2 gt[la,lb] gtu[uc,ud] cdphi2[lc,ld]
                    + 4 cdphi[la] cdphi[lb]
                    - 4 gt[la,lb] gtu[uc,ud] cdphi[lc] cdphi[ld]),

    Atm[ua,lb] -> gtu[ua,uc] At[lc,lb],
    Atu[ua,ub] -> gtu[ub,uc] Atm[ua,lc],

    R[la,lb] -> (+ Rt[la,lb] + Rphi[la,lb]
                 + IfCCZ4[+ 2/phi * (+ g[la,lc] Z[uc] PD[phi,lb]
                                     + g[lb,lc] Z[uc] PD[phi,la]
                                     - g[la,lb] Z[uc] PD[phi,lc])
                          + e4phi Z[uc] PD[gt[la,lb],lc],
                          0]),



    (* Matter *)

    (* rho = n^a n^b T_ab *)
    rho -> IfMatter[1/alpha^2
                    (T00 - 2 beta[ua] T0[la] + beta[ua] beta[ub] T[la,lb]),
                    0],

    (* S_i = -p^a_i n^b T_ab, where p^a_i = delta^a_i + n^a n_i *)
    S[la] -> IfMatter[-1/alpha * (T0[la] - beta[ub] T[la,lb]), 0],

    (* trS = gamma^ij T_ij  *)
    trS -> IfMatter[gu[ua,ub] T[la,lb], 0],



    (* Calculate RHS terms *)

    (* PRD 62 044034 (2000), eqn. (10) *)
    (* PRD 67 084023 (2003), eqns. (16) and (23) *)
    dot[phi] -> (+ IfW[1/3 phi, -1/6] (alpha trK - PD[beta[ua],la])
                 + Upwind[beta[ua], phi, la]
                 + Dissipation[phi]),

    (* PRD 62, 044034 (2000), eqn. (9) *)
    (* gr-qc:1106.2254 (2011), eqn. (14) *)
    (* removing trA from Aij ensures that dot[detgt] = 0 *)
    trAt           -> Atm[ua,la],
    dot[gt[la,lb]] -> (- 2 alpha * (+ At[la,lb]
                                    - IfCCZ4[1/3 gt[la,lb] trAt, 0])
                       + gt[la,lc] PD[beta[uc],lb]
                       + gt[lb,lc] PD[beta[uc],la]
                       - 2/3 gt[la,lb] PD[beta[uc],lc]
                       + Upwind[beta[uc], gt[la,lb], lc]
                       + Dissipation[gt[la,lb]]),

    (* PRD 62 044034 (2000), eqn. (20) *)
    (* PRD 67 084023 (2003), eqn. (26) *)
    (* gr-qc:1106.2254 (2011), eqn. (19) *)
    (* Equation (4.28) in Baumgarte & Shapiro (Phys. Rept. 376 (2003) 41-131) *)
    (* CCZ4: Adding Z terms by changing Xtn to Xt, also adding extra Z
       and Theta terms *)
    dotXt[ua] -> (- 2 Atu[ua,ub] PD[alpha,lb]
                  + 2 alpha * (+ Gt[ua,lb,lc] Atu[ub,uc]
                               - 2/3 gtu[ua,ub] PD[trK,lb]
                               + 6 Atu[ua,ub] cdphi[lb])
                  + gtu[ub,uc] PD[beta[ua],lb,lc]
                  + 1/3 gtu[ua,ub] PD[beta[uc],lb,lc]
                  - Xtn[ub] PD[beta[ua],lb]
                  + 2/3 Xtn[ua] PD[beta[ub],lb]
                  + IfCCZ4[+ GammaShift 2 e4phi (- Z[ub] PD[beta[ua],lb]
                                                 + 2/3 Z[ua] PD[beta[ub],lb])
                           - 4/3 alpha e4phi Z[ua] trK
                           + 2 gtu[ua,ub] (+ alpha PD[Theta,lb]
                                           - Theta PD[alpha,lb])
                           - 2 alpha e4phi dampk1 Z[ua],
                           0]
                  + IfMatter[- 16 Pi alpha gtu[ua,ub] S[lb]]),
    dot[Xt[ua]] -> (+ dotXt[ua]
                    + Upwind[beta[ub], Xt[ua], lb]
                    + Dissipation[Xt[ua]]),

    (* gr-qc:1106.2254 (2011), eqn. (18) *)
    trR      -> gu[ua,ub] R[la,lb],
    dotTheta -> (- PD[alpha,la] Z[ua]
                 - dampk1 (2 + dampk2) alpha Theta
                 + 1/2 alpha * (+ trR
                                - Atm[ua,lb] Atm[ub,la]
                                + 2/3 trK^2
                                - 2 trK Theta)
                 + IfMatter[- 8 Pi alpha rho]),
    dot[Theta] -> IfCCZ4[+ dotTheta
                         + Upwind[beta[ua], Theta, la]
                         + Dissipation[Theta],
                         0],

    (* PRD 62 044034 (2000), eqn. (11) *)
    (* gr-qc:1106.2254 (2011), eqn. (17) *)
    (* Equation (4.21) in Baumgarte & Shapiro (Phys. Rept. 376 (2003) 41-131) *)
    (* CCZ4: Adding the RHS of Theta to K, because K_Z4 = K_BSSN + 2
       Theta. Also adding the Z term, as it has to cancel with the one
       in Theta. *)
    dottrK -> (- em4phi * (+ gtu[ua,ub] (+ PD[alpha,la,lb]
                                         + 2 cdphi[la] PD[alpha,lb])
                           - Xtn[ua] PD[alpha,la])
               + alpha * (Atm[ua,lb] Atm[ub,la] + 1/3 trK^2)
               + IfCCZ4[+ 2 dotTheta + 2 PD[alpha,la] Z[ua]
                        + dampk1 (1 - dampk2) alpha Theta,
                        0]
               + IfMatter[4 Pi alpha * (rho + trS)]),
    dot[trK] -> (+ dottrK
                 + Upwind[beta[ua], trK, la]
                 + Dissipation[trK]),

    (* PRD 62 044034 (2000), eqn. (12) *)
    (* TODO: Should we use the Hamiltonian constraint to make Rij tracefree? *)
    (* gr-qc:1106.2254 (2011), eqn. (15) *)
    (* Equation (4.23) in Baumgarte & Shapiro (Phys. Rept. 376 (2003) 41-131) *)
    (* CCZ4: Adding Z terms in the Ricci and Theta terms *)
    Ats[la,lb]     -> (- CDt[alpha,la,lb] +
                       + 2 (PD[alpha,la] cdphi[lb] + PD[alpha,lb] cdphi[la])
                       + alpha R[la,lb]),
    trAts          -> gu[ua,ub] Ats[la,lb],
    dot[At[la,lb]] -> (+ em4phi * (Ats[la,lb] - 1/3 g[la,lb] trAts)
                       + alpha * (+ (trK - IfCCZ4[2 Theta, 0]) At[la,lb]
                                  - 2 At[la,lc] Atm[uc,lb])
                       + At[la,lc] PD[beta[uc],lb]
                       + At[lb,lc] PD[beta[uc],la]
                       - 2/3 At[la,lb] PD[beta[uc],lc]
                       + Upwind[beta[uc], At[la,lb], lc]
                       + Dissipation[At[la,lb]]
                       + IfMatter[- em4phi alpha 8 Pi
                                  (T[la,lb] - 1/3 g[la,lb] trS)]),

    dotalpha   -> - (harmonicF alpha^harmonicN
                     IfA[A,
                         + trK - IfCCZ4[2 Theta, 0]
                         + alphaDriver (alpha - 1)]),
    dot[alpha] -> (+ dotalpha
                   + IfThen[advectLapse!=0, Upwind[beta[ua], alpha, la], 0]
                   + Dissipation[alpha]),

    dot[A] -> IfA[+ dottrK - IfCCZ4[2 dot[Theta], 0]
                  + IfThen[fixAdvectionTerms!=0, Upwind[beta[ua], trK, la], 0]
                  - (alphaDriver
                     (+ A
                      + IfThen[fixAdvectionTerms!=0 && advectLapse!=0,
                               Upwind[beta[ua], alpha, la] /
                               (- harmonicF alpha^harmonicN),
                               0]))
                  + IfThen[fixAdvectionTerms==0 && advectLapse!=0,
                           Upwind[beta[ua], A, la],
                           0]
                  + Dissipation[A],
                  0],

    betaDriverValue      -> betaDriverFunc[r],
    shiftGammaCoeffValue -> shiftGammaCoeffFunc[r],

    (* TODO COMPATIBILITY: this is zero analytically *)
    ddetgt[la] -> gtu[ub,uc] PD[gt[lb,lc],la],

    dotbeta[ua] -> IfThen[shiftFormulation == shiftGammaDriver,
                          (* Gamma driver *)
                          shiftGammaCoeffValue alpha^shiftAlphaPower
                          IfB[B[ua], Xt[ua] - betaDriverValue beta[ua]],
                          (* harmonic *)
                          - (gu[ua,ub] alpha
                             (+ 2 alpha cdphi[lb] + PD[alpha,lb]
                              + 1/2 alpha ddetgt[lb]
                              - alpha gtu[uc,ud] PD[gt[lb,lc],ld]))],

    (* TODO: add not only advection terms, but full Lie derivatives! *)
    dot[beta[ua]] -> (+ dotbeta[ua]
                      + IfThen[advectShift!=0,
                               Upwind[beta[ub], beta[ua], lb],
                               0]
                      + Dissipation[beta[ua]]),

    dot[B[ua]] -> IfB[+ dotXt[ua]
                      + IfThen[fixAdvectionTerms!=0,
                               Upwind[beta[ub], Xt[ua], lb],
                               0]
                      - (betaDriverValue
                         (+ B[ua]
                          + IfThen[fixAdvectionTerms!=0 && advectShift!=0,
                                   Upwind[beta[ub], beta[ua], lb] /
                                   (shiftGammaCoeffValue alpha^shiftAlphaPower),
                                   0]))
                      + IfThen[fixAdvectionTerms==0 && advectShift!=0,
                               Upwind[beta[ub], B[ua], lb],
                               0]
                      + Dissipation[B[ua]],
                      0],



    (* Calculate ADMBase variables *)
    admg[la,lb]   -> g[la,lb],
    admK[la,lb]   -> e4phi At[la,lb] + 1/3 g[la,lb] trK,
    admalpha      -> alpha,
    admdtalpha    -> (+ dotalpha
                      + IfThen[advectLapse!=0, Upwind[beta[ua], alpha, la], 0]),
    admbeta[ua]   -> beta[ua],
    admdtbeta[ua] -> (+ dotbeta[ua]
                      + IfThen[advectShift!=0,
                               Upwind[beta[ub], beta[ua], lb],
                               0]),



    (* Calculate constraints *)

    (* det gamma-tilde *)
    cS -> 0 (* TODO COMPATIBILITY detgtExpr - 1 *),

    (* Gamma constraint *)
    cXt[ua] -> Xtn[ua] - Xt[ua],

    (* trace A-tilde *)
    cA -> trAt,

    (* H -> trR - Km[ua,lb] Km[ub,la] + trK^2 *)
    (* PRD 67, 084023 (2003), eqn. (19) *)
    H -> trR - Atm[ua,lb] Atm[ub,la] + 2/3 trK^2 - IfMatter[16 Pi rho],

    (* TODO: use PRD 67, 084023 (2003), eqn. (20) *)
    M[la] -> (+ gtu[ub,uc] (CDt[At[la,lb],lc] + 6 At[la,lb] cdphi[lc])
              - 2/3 PD[trK,la]
              - IfMatter[8 Pi S[la]])}};



(******************************************************************************)
(* Actual calculations *)
(******************************************************************************)

(* Initial conditions *)

initialMinkowskiCalcUNUSED = PartialCalculation[
  BSSNFromMinkowskiCalc, "",
  {
    Name                  -> thorn <> "_InitialMinkowski",
    Schedule              -> {"IN ADMBase_InitialData"},
    ConditionalOnKeywords -> {{"evolution_method", thorn}}
  },
  {phi, gt[la,lb], Xt[ua], trK, At[la,lb], IfCCZ4[Theta],
   alpha, IfA[A], beta[ua], IfB[B[ua]]}];



initialADMBase1EverywhereCalc = PartialCalculation[
  BSSNFromADMBaseCalc, "",
  {
    Name                  -> thorn <> "_InitialADMBase1Everywhere",
    Schedule              -> {"AT initial " <>
                              "AFTER ADMBase_PostInitial"},
    ConditionalOnKeywords -> {{"evolution_method", thorn}}
  },
  {phi, gt[la,lb], trK, At[la,lb], IfCCZ4[Theta], alpha, beta[ua]}];

initialADMBase2InteriorCalc = PartialCalculation[
  BSSNFromADMBaseCalc, "",
  {
    Name                  -> thorn <> "_InitialADMBase2Interior",
    Schedule              -> {"AT initial " <>
                              "AFTER ADMBase_PostInitial " <>
                              "AFTER " <> thorn <> "_InitialADMBase1Everywhere"},
    ConditionalOnKeywords -> {{"evolution_method", thorn}},
    Where                 -> Interior
  },
  {Xt[ua], IfA[A], IfB[B[ua]]}];

(* Boundary conditions can also be applied via ML_BSSN_Helper *)
(* TODO: Check CaKernel branch for Kranc-only solution *)
initialADMBase2BoundaryScalarCalc = {
  Name                  -> thorn <> "_InitialADMBase2BoundaryScalar",
  Schedule              -> {"AT initial " <>
                            "AFTER ADMBase_PostInitial"},
  ConditionalOnKeywords -> {{"evolution_method", thorn},
                            {"initial_boundary_condition", "scalar"}},
  Where                 -> BoundaryNoSync, (* evolution BC will sync *)
  Equations -> {
    Xt[ua] -> 0,
    A      -> 0 // IfA,
    B[ua]  -> 0 // IfB}};



(* Enforce algebraic BSSN constraints *)

enforceEverywhereCalc = PartialCalculation[
  BSSNToBSSNCalc, "",
  {
    Name                  -> thorn <> "_EnforceEverywhere",
    Schedule              -> {"IN MoL_PostStepModify"},
    ConditionalOnKeywords -> {{"evolution_method", thorn}}
  },
  {gt[la,lb], At[la,lb], alpha}];



(* Calculate ADMBase variables *)

ADMBaseEverywhereCalc = PartialCalculation[
  EverythingFromBSSNCalc, "",
  {
    Name                  -> thorn <> "_ADMBaseEverywhere",
    Schedule              -> {"IN MoL_PostStep " <>
                              "AFTER " <> thorn <> "_ApplyBCs " <>
                              "BEFORE ADMBase_SetADMVars"},
    ConditionalOnKeywords -> {{"evolution_method", thorn}}
  },
  {admg[la,lb], admK[la,lb], admalpha, admbeta[ua]}];

ADMBaseInteriorCalc = PartialCalculation[
  EverythingFromBSSNCalc, "",
  {
    Name                  -> thorn <> "_ADMBaseInterior",
    Schedule              -> {"IN MoL_PostStep " <>
                              "AFTER " <> thorn <> "_ApplyBCs " <>
                              "BEFORE ADMBase_SetADMVars"},
    ConditionalOnKeywords -> {{"evolution_method", thorn}},
    Where                 -> InteriorNoSync
  },
  {admdtalpha, admdtbeta[ua]}];

(* TODO: offer other boundary conditions, e.g. extrapolation *)
(* TODO: add symmetry conditions via Kranc *)
ADMBaseADMBaseBoundaryScalarCalc = {
  Name                  -> thorn <> "_ADMBaseBoundaryScalar",
  Schedule              -> {"IN MoL_PostStep " <>
                            "AFTER " <> thorn <> "_ApplyBCs " <>
                            "AFTER " <> thorn <> "_ADMBaseInterior " <>
                            "BEFORE ADMBase_SetADMVars"},
  ConditionalOnKeywords -> {{"evolution_method", thorn}},
  Where                 -> Boundary,
  Equations -> {
    admdtalpha    -> 0,
    admdtbeta[ua] -> 0}};



(* Calculate RHS variables *)
(* TODO: Move this to MoL_PostStep *)

evolutionInteriorCalc = PartialCalculation[
  EverythingFromBSSNCalc, "",
  {
    Name                  -> thorn <> "_EvolutionInterior",
    Schedule              -> {"IN MoL_CalcRHS"},
    ConditionalOnKeywords -> {{"evolution_method", thorn},
                              {"rhs_evaluation", "combined"}},
    Where                 -> InteriorNoSync (* RHS is not sync'ed *)
  },
  {dot[phi], dot[gt[la,lb]], dot[Xt[ua]], dot[trK], dot[At[la,lb]],
   IfCCZ4[dot[Theta]],
   dot[alpha], IfA[dot[A]], dot[beta[ua]], IfB[dot[B[ua]]]}];

evolutionInteriorCalc1 = PartialCalculation[
  evolutionInteriorCalc, "1",
  {
    ConditionalOnKeywords -> {{"evolution_method", thorn},
                              {"rhs_evaluation", "split3"}}
  },
  {dot[phi], dot[gt[la,lb]], dot[alpha], IfA[dot[A]]}];

evolutionInteriorCalc2 = PartialCalculation[
  evolutionInteriorCalc, "2",
  {
    ConditionalOnKeywords -> {{"evolution_method", thorn},
                              {"rhs_evaluation", "split3"}}
  },
  {dot[Xt[ua]], dot[trK], IfCCZ4[dot[Theta]], dot[beta[ua]], IfB[dot[B[ua]]]}];

evolutionInteriorCalc3 = PartialCalculation[
  evolutionInteriorCalc, "3",
  {
    ConditionalOnKeywords -> {{"evolution_method", thorn},
                              {"rhs_evaluation", "split3"}}
  },
  {dot[At[la,lb]]}];

evolutionInteriorCalcSplitBy = PartialCalculation[
  EverythingFromBSSNCalc, "",
  {
    Name                  -> thorn <> "_EvolutionInteriorSplitBy",
    Schedule              -> {"IN MoL_CalcRHS"},
    ConditionalOnKeywords -> {{"evolution_method", thorn},
                              {"rhs_evaluation", "splitBy"}},
    Where                 -> InteriorNoSync, (* RHS is not sync'ed *)
    SplitBy -> {
      {dot[phi], dot[gt[la,lb]], dot[alpha], IfA[dot[A]]},
      {dot[Xt[ua]], dot[trK], IfCCZ4[dot[Theta]],
       dot[beta[ua]], IfB[dot[B[ua]]]},
      {dot[At[la,lb]]}}
  },
  {dot[phi], dot[gt[la,lb]], dot[Xt[ua]], dot[trK], dot[At[la,lb]],
   IfCCZ4[dot[Theta]],
   dot[alpha], IfA[dot[A]], dot[beta[ua]], IfB[dot[B[ua]]]}];

evolutionInteriorCalcSeparatedDerivatives_UNUSED = PartialCalculation[
  EverythingFromBSSNCalc, "",
  {
    Name                  -> thorn <> "_EvolutionInteriorSeparatedDerivatives",
    Schedule              -> {"IN MoL_CalcRHS"},
    ConditionalOnKeywords -> {{"evolution_method", thorn},
                              {"rhs_evaluation", "separatedDerivatives"}},
    Where                 -> InteriorNoSync, (* RHS is not sync'ed *)
    SeparatedDerivatives  -> {PD[_,i_] | PD[_,i_,i_]},
    SeparatedDerivatives2 -> {PD[_,i_,j_] /; i!=j}
    (*
    SeparatedDerivatives  -> {PD[_,i_,j_] /; i!=j}
    *)
  },
  {dot[phi], dot[gt[la,lb]], dot[Xt[ua]], dot[trK], dot[At[la,lb]],
   IfCCZ4[dot[Theta]],
   dot[alpha], IfA[dot[A]], dot[beta[ua]], IfB[dot[B[ua]]]}];

(* Boundary conditions can also be applied via ML_BSSN_Helper *)
(* TODO: Check CaKernel branch for Kranc-only solution *)
evolutionBoundaryScalarCalc = {
  Name                  -> thorn <> "_EvolutionBoundaryScalar",
  Schedule              -> {"IN MoL_CalcRHS"},
  ConditionalOnKeywords -> {{"evolution_method", thorn},
                            {"rhs_boundary_condition", "scalar"}},
  Where                 -> BoundaryNoSync, (* RHS is not sync'ed *)
  Equations -> {
    dot[phi]       -> 0,
    dot[gt[la,lb]] -> 0,
    dot[Xt[ua]]    -> 0,
    dot[Theta]     -> 0 // IfCCZ4,
    dot[trK]       -> 0,
    dot[At[la,lb]] -> 0,
    dot[alpha]     -> 0,
    dot[A]         -> 0 // IfA,
    dot[beta[ua]]  -> 0,
    dot[B[ua]]     -> 0 // IfB}};

(* Initialise the RHS variables in analysis in case they are going to
   be output. This is just for cosmetic reasons. We cannot set just
   the ghost zones, so we set all points to zero. *)
(* TODO: offer this in CalcRHS as well, at least enabled via a
   parameter, or better by checking MoL::init_RHS_zero *)
evolutionAnalysisInitCalc =
{
  Name                  -> thorn <> "_EvolutionAnalysisInit",
  Schedule              -> {"IN " <> thorn <> "_EvolutionAnalysis"},
  Before                -> {thorn <> "_EvolutionAnalysisInterior"},
  ConditionalOnKeywords -> {{"evolution_method", thorn}},
  Equations -> {
    dot[phi]       -> 0,
    dot[gt[la,lb]] -> 0,
    dot[Xt[ua]]    -> 0,
    dot[Theta]     -> 0 // IfCCZ4,
    dot[trK]       -> 0,
    dot[At[la,lb]] -> 0,
    dot[alpha]     -> 0,
    dot[A]         -> 0 // IfA,
    dot[beta[ua]]  -> 0,
    dot[B[ua]]     -> 0 // IfB}};

evolutionAnalysisInteriorCalc = PartialCalculation[
  evolutionInteriorCalc, "",
  {
    Name                  -> thorn <> "_EvolutionAnalysisInterior",
    Schedule              -> {"IN " <> thorn <> "_EvolutionAnalysis"},
    ConditionalOnKeywords -> {{"evolution_method", thorn}},
    (* TODO: sync and apply boundary conditions,
       but add a parameter for testsuite backward compatibility *)
    Where                 -> InteriorNoSync
  },
  {dot[phi], dot[gt[la,lb]], dot[Xt[ua]], dot[trK], dot[At[la,lb]],
   IfCCZ4[dot[Theta]],
   dot[alpha], IfA[dot[A]], dot[beta[ua]], IfB[dot[B[ua]]]}];

(* We don't need to apply a scalar boundary condition since the RHS
   has already been set to zero everywhere. *)



(* Calculate constraints *)

constraintsEverywhereCalc = PartialCalculation[
  EverythingFromBSSNCalc, "",
  {
    Name                  -> thorn <> "_ConstraintsEverywhere",
    Schedule              -> Automatic,
    ConditionalOnKeywords -> {{"evolution_method", thorn}},
    Where                 -> Interior (* TODO COMPATIBILITY Everywhere *)
  },
  {cS, cA}];

constraintsInteriorCalc = PartialCalculation[
  EverythingFromBSSNCalc, "",
  {
    Name                  -> thorn <> "_ConstraintsInterior",
    Schedule              -> Automatic,
    ConditionalOnKeywords -> {{"evolution_method", thorn}},
    Where                 -> Interior
  },
  {cXt[ua], H, M[la]}];



(******************************************************************************)
(* Construct the thorn *)
(******************************************************************************)

calculations = {
  initialADMBase1EverywhereCalc,
  initialADMBase2InteriorCalc,
  initialADMBase2BoundaryScalarCalc,
  enforceEverywhereCalc,
  ADMBaseEverywhereCalc,
  ADMBaseInteriorCalc,
  ADMBaseADMBaseBoundaryScalarCalc,
  evolutionInteriorCalc,
  (* evolutionInteriorCalc1, evolutionInteriorCalc2, evolutionInteriorCalc3, *)
  evolutionInteriorCalcSplitBy,
  (* evolutionInteriorCalcSeparatedDerivatives, *)
  evolutionBoundaryScalarCalc,
  evolutionAnalysisInitCalc,
  evolutionAnalysisInteriorCalc,
  constraintsEverywhereCalc,
  constraintsInteriorCalc};

CreateKrancThornTT[
  groups, ".", thorn,
  Calculations -> calculations,
  DeclaredGroups -> declaredGroupNames,
  PartialDerivatives -> derivatives,
  InheritedImplementations -> inheritedImplementations,
  ExtendedKeywordParameters -> extendedKeywordParameters,
  KeywordParameters -> keywordParameters,
  IntParameters -> intParameters,
  RealParameters -> realParameters,
  EvolutionTimelevels -> maxTimelevels,
  DefaultEvolutionTimelevels -> Min[3,maxTimelevels],
  UseJacobian -> True,
  UseLoopControl -> True,
  UseOpenCL -> useOpenCL,
  UseVectors -> useVectors];
