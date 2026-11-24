# =============================================================================
# Teleconsultas SUS-RJ – PREDIÇÃO DE COMPARECIMENTO – versão final
# PUC-Rio, IND2622 – Adriana Nascimento, Daniel Araújo, Felipe Abuzed
#
# Base: aba "Base final tratada" (fonte única: nenhuma correção, filtro ou variável nova no código).
# Alvo: Compareceu = 1 (Executado) | Não compareceu = 0 (Faltou + Cancelado pelo Paciente + Cancelado pelo Profissional).
# Células com erro #N/D (PROCV que não achou o bairro no de-para) são lidas como vazias e rotuladas "Bairro fora do de-para".
# Território: usamos REGIÃO (de residência e de atendimento), e não a área programática.
# =============================================================================

# %% ETAPA 0 – CONFIGURAÇÃO -----------------------------------------------------
CAMINHO_PLANILHA = "SaudeDigital_agosto2026_dados_a_serem_trabalhados_v4.xlsx"
PASTA_SAIDA = "resultados_final"
MODELOS_A_RODAR = ["RegressaoLogistica", "RandomForest", "XGBoost"]
PROPORCAO_TREINO = 0.80
SEMENTE = 42
LIMITE_GAP_AUC = 0.10
DIFERENCA_C2_C4 = 0.01
MIN_PERFIL = 30

# %% ETAPA 1 – BIBLIOTECAS ------------------------------------------------------
import os, re, json, unicodedata, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score, average_precision_score
from xgboost import XGBClassifier
import shap

warnings.filterwarnings("ignore")
os.makedirs(PASTA_SAIDA, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
AZUL, LARANJA, VERDE, CINZA, VERMELHO = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a8a", "#C44E52"
NOME = {"RegressaoLogistica": "Regressão Logística", "RandomForest": "Random Forest", "XGBoost": "XGBoost"}
COR = {"RegressaoLogistica": AZUL, "RandomForest": LARANJA, "XGBoost": VERDE}
br = lambda v, d=3: f"{v:.{d}f}".replace(".", ",")
pct = lambda v: f"{v:.1%}".replace(".", ",")

def norm(s):
    if pd.isna(s):
        return None
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", s).strip()

# %% ETAPA 2 – CARGA E NOMES CURTOS ---------------------------------------------
NOMES = {"area_programatica": "ap", "Região de atendimento": "regiao_atendimento", "sexo": "sexo",
         "raca_cor": "raca_cor", "bairro_residencia": "bairro",
         "Região (trazer via tabela de-para área programática ou bairro de residência?)": "regiao_residencia",
         "Renda média por família (nova base para cruzar)": "renda_media", "Classe associada": "classe",
         "Faixa salário mínimo": "faixa_sm", "profissional_cbo_descricao": "profissional",
         "dthr_marcacao": "dthr_marcacao", "estado_marcacao": "estado_marcacao",
         "Nome da doença": "nome_doenca", "mes_ano_nascimento": "nascimento", "Faixa etária": "faixa_etaria"}
base = pd.read_excel(CAMINHO_PLANILHA, sheet_name="Base final tratada").rename(columns=NOMES)
renda_tab = pd.read_excel(CAMINHO_PLANILHA, sheet_name="Renda média")
print(f"[2] Base final tratada v3: {len(base)} linhas")

# %% ETAPA 3 – ALVO (a aba tratada é a fonte única de verdade) ----------------------
fluxo = [("Base final tratada (todas as linhas entram)", len(base))]
n_prof = int((base["estado_marcacao"] == "Cancelado pelo Profissional").sum())
ESTADOS = ["Executado", "Faltou", "Cancelado pelo Paciente", "Cancelado pelo Profissional"]
dados = base[base["estado_marcacao"].isin(ESTADOS)].copy()
fluxo.append((f"Base de modelagem ({', '.join(e for e in ESTADOS if e in set(dados['estado_marcacao']))})", len(dados)))
dados["compareceu"] = (dados["estado_marcacao"] == "Executado").astype(int)
composicao = (dados["estado_marcacao"].value_counts().rename_axis("estado na base").reset_index(name="agendamentos"))
composicao["classe do alvo"] = np.where(composicao["estado na base"] == "Executado", "Compareceu", "Não compareceu")
composicao["% da base"] = composicao["agendamentos"] / len(dados)
# Região de residência vazia: sinalizada (não corrigida)
sem_regiao = dados.loc[dados["regiao_residencia"].isna(), ["ID", "bairro", "estado_marcacao"]].sort_values("ID")
print(f"[3] Região de residência vazia ou com #N/D: {len(sem_regiao)} linhas" +
      (f". IDs: {sem_regiao['ID'].tolist()[:15]} ..." if len(sem_regiao) else " (coluna completa)"))
print(f"[3] Cancelado pelo Profissional na base: {n_prof}")
NA = "Bairro fora do de-para"   # célula com erro #N/D (PROCV sem correspondência): não é corrigida, só rotulada
for c in ["regiao_residencia", "classe", "faixa_sm"]:
    dados[c] = dados[c].fillna(NA)
media = dados["compareceu"].mean()
print(f"[3] Base de modelagem: {len(dados)} | Compareceu {dados['compareceu'].sum()} ({media:.1%}) | "
      f"Não compareceu {(1 - dados['compareceu']).sum()} ({1 - media:.1%})")

# %% ETAPA 4 – DETECTOR DE VAZAMENTO (só diagnóstico: as colunas com vazamento ficam fora dos cenários) -------------------------------------------------
CAND = {"sexo": "sexo", "raca_cor": "raça/cor", "faixa_etaria": "faixa etária", "bairro": "bairro",
        "regiao_residencia": "região de residência", "renda_media": "renda média", "classe": "classe",
        "regiao_atendimento": "região de atendimento", "profissional": "profissional (CBO)",
        "tipo_consulta": "tipo de consulta", "eh_coleta": "coleta", "dthr_inicio_atendimento": "início do atendimento",
        "Mês e ano da consulta": "mês e ano da consulta", "condicoes_ativas": "CID (condições ativas)",
        "nome_doenca": "nome da doença"}
vaz = []
for c, rot in CAND.items():
    s = dados[c].astype(str)
    v = dados[c].isna() | s.isin(["[]", "nan", "Sem informação", "Bairro fora do de-para", "n/a-n/a", "Nenhum diagnóstico registrado"])
    a, b = v[dados["compareceu"] == 1].mean(), v[dados["compareceu"] == 0].mean()
    vaz.append({"variável": rot, "% vazio entre quem compareceu": round(a, 3), "% vazio entre quem não compareceu": round(b, 3),
                "diferença (p.p.)": round(abs(a - b) * 100, 1), "vazamento?": "SIM" if abs(a - b) > 0.30 else "não"})
vazamento = pd.DataFrame(vaz)

# %% ETAPA 5 – Q1 E Q2 (a morbidade saiu do estudo: é registrada depois da consulta) ------------------------------------------------------------
Q1_VARS = {"sexo": "Sexo", "raca_cor": "Raça/cor", "faixa_etaria": "Faixa etária",
           "regiao_residencia": "Região de residência", "regiao_atendimento": "Região de atendimento"}
q1 = []
for v, rot in Q1_VARS.items():
    g = dados.groupby(v)["compareceu"].agg(agendamentos="size", compareceram="sum").reset_index()
    g.insert(0, "variável", rot)
    q1.append(g.rename(columns={v: "categoria"}))
q1 = pd.concat(q1)
q1["taxa"] = q1["compareceram"] / q1["agendamentos"]

ORD_CL = ["C2", "C1", "B2", "B1", "A", "Bairro fora do de-para"]
q2c = dados.groupby("classe")["compareceu"].agg(agendamentos="size", compareceram="sum").reindex(ORD_CL).dropna().reset_index()
q2c["renda média do grupo (R$)"] = dados.groupby("classe")["renda_media"].mean().reindex(q2c["classe"]).round(0).values
q2c["taxa"] = q2c["compareceram"] / q2c["agendamentos"]
cr = dados.dropna(subset=["renda_media"]).copy()
cr["quintil"] = pd.qcut(cr["renda_media"].rank(method="first"), 5, labels=["1º quintil (20% mais pobres)", "2º quintil", "3º quintil", "4º quintil", "5º quintil (20% mais ricos)"])
q2q = cr.groupby("quintil")["compareceu"].agg(agendamentos="size", compareceram="sum").reset_index()
q2q["renda média do quintil (R$)"] = cr.groupby("quintil")["renda_media"].mean().round(0).values
q2q["taxa"] = q2q["compareceram"] / q2q["agendamentos"]

def erreygers(y, r):
    pos = pd.Series(r).rank(method="average").values / len(r)
    return 8 * np.cov(y, pos, bias=True)[0, 1]
rng = np.random.default_rng(SEMENTE)
yv, rv = cr["compareceu"].values, cr["renda_media"].values
E = erreygers(yv, rv)
ic_E = np.percentile([erreygers(yv[i], rv[i]) for i in (rng.integers(0, len(yv), len(yv)) for _ in range(1000))], [2.5, 97.5])
E_reg = {}
for reg in ["Zona norte", "Zona oeste"]:
    s = cr[cr["regiao_residencia"] == reg]
    E_reg[reg] = erreygers(s["compareceu"].values, s["renda_media"].values)

from scipy.stats import chi2_contingency
testes = []
for v, rot in {**Q1_VARS, "classe": "Classe ABEP"}.items():
    ct = pd.crosstab(dados[v], dados["compareceu"])
    chi2, pval, gl, _ = chi2_contingency(ct)
    cramer = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    testes.append({"variável": rot, "qui-quadrado": round(chi2, 1), "graus de liberdade": gl,
                   "p-valor": float(f"{pval:.3g}"), "V de Cramér": round(cramer, 3),
                   "força da associação": "forte" if cramer >= 0.3 else ("moderada" if cramer >= 0.1 else "fraca")})
testes = pd.DataFrame(testes).sort_values("V de Cramér", ascending=False)
print(f"[5] Erreygers {E:.4f} (IC {ic_E[0]:.4f} a {ic_E[1]:.4f})")
print(testes.to_string(index=False))

# %% ETAPA 6 – CENÁRIOS ---------------------------------------------------------------
CENARIOS = {
    "C1": ["sexo", "raca_cor", "faixa_etaria"],
    "C2": ["sexo", "raca_cor", "faixa_etaria", "regiao_residencia", "classe", "renda_media"],
    "C3": ["sexo", "raca_cor", "faixa_etaria", "regiao_residencia", "classe", "renda_media", "regiao_atendimento"],
    "C4": ["sexo", "raca_cor", "faixa_etaria", "regiao_residencia", "classe", "renda_media", "regiao_atendimento", "bairro"],
}
EXTRAS = {}          # o estudo final usa só os conjuntos C1 a C4
TODOS = {**CENARIOS, **EXTRAS}
dados["ap"] = dados["ap"].astype(str)
NUM = ["renda_media"]
JUSTIFICATIVA = pd.DataFrame([
    ["C1 – Pessoa", "sexo, raça/cor, faixa etária",
     "São as características da própria pessoa (fatores predisponentes do modelo de Andersen). Respondem à Q1.",
     "Renda individual: não existe na base."],
    ["C2 – + Território e renda", "C1 + região de residência, classe ABEP, renda média do bairro",
     "Onde a pessoa mora e a condição socioeconômica do bairro (determinantes sociais, Solar & Irwin). Respondem à Q2.",
     "Faixa de salário mínimo: é a mesma informação da classe, com outro nome."],
    ["C3 – + Região de atendimento", "C2 + região da unidade de atendimento",
     "Onde o serviço é oferecido (dimensão de oferta, Levesque). Usamos a região, mais fácil de interpretar que a AP.",
     "Área programática: o grupo decidiu apresentar o território por região."],
    ["C4 – + Bairro", "C3 + bairro de residência",
     "Detalha o território dentro de cada região.",
     "Dia, turno e mês: cada unidade tem horário próprio. Tipo de consulta, coleta, datas de atendimento, CID e morbidade: vazamento."],
], columns=["cenário", "variáveis", "por que entram", "o que ficou de fora e por quê"])

# %% ETAPA 7 – DIVISÃO TEMPORAL 80/20 -------------------------------------------------
dados = dados.sort_values("dthr_marcacao").reset_index(drop=True)
n_tr = int(len(dados) * PROPORCAO_TREINO)
treino, teste = dados.iloc[:n_tr], dados.iloc[n_tr:]
y_tr, y_te = treino["compareceu"], teste["compareceu"]
CORTE = teste["dthr_marcacao"].min()
print(f"[7] Corte {CORTE:%d/%m/%Y} | treino {len(treino)} ({y_tr.mean():.1%}) | teste {len(teste)} ({y_te.mean():.1%})")

# %% ETAPA 8 – PIPELINE E MODELOS ------------------------------------------------------
def montar(cols, mod, bal, y_ref):
    num = [c for c in cols if c in NUM]
    cat = [c for c in cols if c not in NUM]
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Sem informação")),
                          ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                               sparse_output=False))]), cat)])
    peso = (y_ref == 0).sum() / (y_ref == 1).sum() if bal else 1
    m = {"RegressaoLogistica": LogisticRegression(max_iter=3000, class_weight="balanced" if bal else None),
         "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=10, n_jobs=-1, random_state=SEMENTE,
                                                class_weight="balanced" if bal else None),
         "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                  eval_metric="logloss", random_state=SEMENTE, scale_pos_weight=peso)}[mod]
    return Pipeline([("prep", prep), ("modelo", m)])

def metr(y, p):
    pred = (p >= 0.5).astype(int)
    return (accuracy_score(y, pred), recall_score(y, pred, zero_division=0), roc_auc_score(y, p),
            average_precision_score(y, p))      # PR-AUC: área sob a curva de precisão e revocação

# %% ETAPA 9 – EXPERIMENTAÇÃO -----------------------------------------------------------
linhas, pipes = [], {}
rodadas = [(c, m, b) for c in CENARIOS for m in MODELOS_A_RODAR for b in (False, True)]
rodadas += [(c, m, True) for c in EXTRAS for m in MODELOS_A_RODAR]
for cen, mod, bal in rodadas:
    cols = TODOS[cen]
    p = montar(cols, mod, bal, y_tr).fit(treino[cols], y_tr)
    a1, r1, u1, pr1 = metr(y_tr, p.predict_proba(treino[cols])[:, 1])
    a2, r2, u2, pr2 = metr(y_te, p.predict_proba(teste[cols])[:, 1])
    rid = f"{cen}|{mod}|{'bal' if bal else 'sem'}"
    pipes[rid] = (p, cols)
    linhas.append({"cenário": cen, "modelo": NOME[mod], "balanceado": "sim" if bal else "não",
                   "acurácia treino": a1, "acurácia teste": a2, "recall treino": r1, "recall teste": r2,
                   "AUC treino": u1, "AUC teste": u2, "PR-AUC treino": pr1, "PR-AUC teste": pr2,
                   "gap AUC": u1 - u2, "id": rid})
    print(f"[9] {rid:36s} AUC {u1:.3f} / {u2:.3f} | recall {r2:.3f} | acur {a2:.3f}")
tab = pd.DataFrame(linhas)
tab["elegível"] = np.where(tab["cenário"].isin(CENARIOS) & (tab["balanceado"] == "sim") &
                           (tab["gap AUC"] <= LIMITE_GAP_AUC), "sim", "não")
num_cols = ["acurácia treino", "acurácia teste", "recall treino", "recall teste", "AUC treino", "AUC teste",
            "PR-AUC treino", "PR-AUC teste", "gap AUC"]
tab[num_cols] = tab[num_cols].round(3)

comp = []
for mod in MODELOS_A_RODAR:
    t = tab[(tab["modelo"] == NOME[mod]) & (tab["elegível"] == "sim")].sort_values("AUC teste", ascending=False)
    b = t.iloc[0]
    c2 = tab[(tab["id"] == f"C2|{mod}|bal")].iloc[0]
    c4 = tab[(tab["id"] == f"C4|{mod}|bal")].iloc[0]
    comp.append({"modelo": NOME[mod], "melhor cenário": b["cenário"], "AUC teste (melhor)": b["AUC teste"],
                 "recall teste (melhor)": b["recall teste"], "acurácia teste (melhor)": b["acurácia teste"],
                 "gap AUC (melhor)": b["gap AUC"], "PR-AUC teste (melhor)": b["PR-AUC teste"], "AUC teste C2": c2["AUC teste"], "AUC teste C4": c4["AUC teste"],
                 "C4 − C2": round(c4["AUC teste"] - c2["AUC teste"], 3), "id": b["id"]})
comp = pd.DataFrame(comp).sort_values("AUC teste (melhor)", ascending=False)
vencedor = comp.iloc[0]
mod_v = vencedor["id"].split("|")[1]
# Regra C2 x C4: se o melhor C2 (entre os 3 modelos) ficar a até 0,01 do melhor resultado geral, recomendamos o C2
c2s = tab[(tab["cenário"] == "C2") & (tab["elegível"] == "sim")].sort_values("AUC teste", ascending=False)
melhor_c2 = c2s.iloc[0]
recomendado = melhor_c2["id"] if vencedor["AUC teste (melhor)"] - melhor_c2["AUC teste"] <= DIFERENCA_C2_C4 else vencedor["id"]
mod_v = recomendado.split("|")[1]
comp["motivo"] = ["Maior AUC no teste entre as rodadas elegíveis" if i == 0 else
                  f"AUC {br(vencedor['AUC teste (melhor)'] - r['AUC teste (melhor)'])} abaixo do vencedor"
                  for i, (_, r) in enumerate(comp.iterrows())]
print(f"[9] Vencedor: {vencedor['id']} | Recomendado: {recomendado}")

# %% ETAPA 10 – VALIDAÇÃO -----------------------------------------------------------------
pipe_r, cols_r = pipes[recomendado]
cen_r, _, _ = recomendado.split("|")
prob = pipe_r.predict_proba(teste[cols_r])[:, 1]
auc_r = roc_auc_score(y_te, prob)
val = [{"teste": "a) Chute: dizer sempre 'não compareceu'", "AUC": 0.5, "acurácia": accuracy_score(y_te, np.zeros(len(y_te))),
        "o que prova": "É a linha de base que qualquer modelo precisa vencer."}]
aucs_emb = []
for _ in range(5):                       # 5 sorteios diferentes do alvo
    y_emb = pd.Series(rng.permutation(y_tr.values), index=y_tr.index)
    p_emb = montar(cols_r, mod_v, True, y_emb).fit(treino[cols_r], y_emb).predict_proba(teste[cols_r])[:, 1]
    aucs_emb.append(roc_auc_score(y_te, p_emb))
val.append({"teste": "b) Rótulos embaralhados (média de 5 sorteios)", "AUC": float(np.mean(aucs_emb)), "acurácia": np.nan,
            "o que prova": f"Com o alvo sorteado o modelo não acerta (AUC de {br(min(aucs_emb))} a {br(max(aucs_emb))}): "
                           "o sinal encontrado não é acaso."})
bs = []
for _ in range(1000):
    i = rng.integers(0, len(y_te), len(y_te))
    bs.append(roc_auc_score(y_te.iloc[i], prob[i]))
ic_auc = np.percentile(bs, [2.5, 97.5])
val.append({"teste": "c) Modelo recomendado no teste", "AUC": auc_r,
            "acurácia": accuracy_score(y_te, (prob >= 0.5).astype(int)),
            "o que prova": f"Intervalo de 95% da AUC: {br(ic_auc[0])} a {br(ic_auc[1])}."})
cv = []
for k, (i1, i2) in enumerate(TimeSeriesSplit(n_splits=5).split(dados)):
    a, b = dados.iloc[i1], dados.iloc[i2]
    pp = montar(cols_r, mod_v, True, a["compareceu"]).fit(a[cols_r], a["compareceu"]).predict_proba(b[cols_r])[:, 1]
    cv.append({"janela": k + 1, "período de teste": f"{b['dthr_marcacao'].min():%m/%Y} a {b['dthr_marcacao'].max():%m/%Y}",
               "taxa real de comparecimento": round(b["compareceu"].mean(), 3), "AUC": round(roc_auc_score(b["compareceu"], pp), 3)})
cv = pd.DataFrame(cv)
val.append({"teste": "d) Validação cruzada no tempo (5 janelas)", "AUC": cv["AUC"].mean(), "acurácia": np.nan,
            "o que prova": f"Funciona em períodos diferentes: AUC de {br(cv['AUC'].min())} a {br(cv['AUC'].max())}."})
t_id = tab.set_index("id")
val.append({"teste": "f) Só a pessoa (C1)", "AUC": t_id.loc[f"C1|{mod_v}|bal", "AUC teste"],
            "acurácia": t_id.loc[f"C1|{mod_v}|bal", "acurácia teste"],
            "o que prova": "Sexo, raça/cor e idade sozinhos não preveem o comparecimento."})
validacao = pd.DataFrame(val)
validacao[["AUC", "acurácia"]] = validacao[["AUC", "acurácia"]].round(3)
cal = pd.DataFrame({"p": prob, "y": y_te.values})
cal["faixa"] = pd.qcut(cal["p"].rank(method="first"), 10, labels=False) + 1
calib = cal.groupby("faixa").agg(**{"probabilidade média prevista": ("p", "mean"), "taxa real": ("y", "mean"),
                                    "agendamentos": ("y", "size")}).reset_index().round(3)
print(validacao[["teste", "AUC", "acurácia"]].to_string(index=False)); print(cv.to_string(index=False))

# %% ETAPA 11 – IMPORTÂNCIA ---------------------------------------------------------------
ROT = {"sexo": "Sexo", "raca_cor": "Raça/cor", "faixa_etaria": "Faixa etária", "regiao_residencia": "Região de residência",
       "classe": "Classe ABEP", "renda_media": "Renda média do bairro", "regiao_atendimento": "Região de atendimento",
       "bairro": "Bairro"}
perm = []
for mod in MODELOS_A_RODAR:
    p, c = pipes[f"{cen_r}|{mod}|bal"]
    r = permutation_importance(p, teste[c], y_te, scoring="roc_auc", n_repeats=5, random_state=SEMENTE, n_jobs=-1)
    for j, col in enumerate(c):
        perm.append({"modelo": NOME[mod], "variável": ROT[col], "queda na AUC ao embaralhar": round(r.importances_mean[j], 4)})
perm = pd.DataFrame(perm)
limpar = lambda n: (n.replace("num__", "").replace("cat__", "").replace("infrequent_sklearn", "outros (raros)")
                     .replace("regiao_residencia_", "região de residência: ").replace("regiao_atendimento_", "região de atendimento: ")
                     .replace("faixa_etaria_", "faixa etária: ").replace("raca_cor_", "raça/cor: ").replace("classe_", "classe ")
                     .replace("sexo_", "sexo: ").replace("bairro_", "bairro: ").replace("renda_media", "renda média do bairro"))
det = []
for mod in MODELOS_A_RODAR:
    p, c = pipes[f"{cen_r}|{mod}|bal"]
    nomes = [limpar(n) for n in p.named_steps["prep"].get_feature_names_out()]
    m = p.named_steps["modelo"]
    if mod == "RegressaoLogistica":
        d = pd.DataFrame({"categoria": nomes, "valor": np.exp(m.coef_[0]), "medida": "razão de chances de COMPARECER"})
        d["ord"] = np.abs(np.log(d["valor"]))
    else:
        d = pd.DataFrame({"categoria": nomes, "valor": m.feature_importances_, "medida": "importância nativa"})
        d["ord"] = d["valor"]
    d.insert(0, "modelo", NOME[mod])
    det.append(d.sort_values("ord", ascending=False).head(15).drop(columns="ord"))
detalhe = pd.concat(det).round(4)

# %% ETAPA 11B – VALORES SHAP DO MODELO RECOMENDADO (Lundberg e Lee, 2017) ---------------
prep_r, modelo_r = pipe_r.named_steps["prep"], pipe_r.named_steps["modelo"]
X_te_r = prep_r.transform(teste[cols_r])
if mod_v == "RegressaoLogistica":
    explicador = shap.LinearExplainer(modelo_r, prep_r.transform(treino[cols_r]))
else:
    explicador = shap.TreeExplainer(modelo_r)                # TreeSHAP: valores exatos para árvores
sv = explicador.shap_values(X_te_r)
sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if np.ndim(sv) == 3 else sv)
# cada variável categórica vira várias colunas 0/1: somamos o SHAP dessas colunas por variável
ordem_cols = sorted(cols_r, key=len, reverse=True)
var_col = [next(c for c in ordem_cols if n.split("__", 1)[1].startswith(c)) for n in prep_r.get_feature_names_out()]
shap_var = pd.DataFrame(sv, columns=var_col).T.groupby(level=0).sum().T
shap_imp = shap_var.abs().mean().sort_values(ascending=False)
shap_tab = pd.DataFrame({"variável": [ROT[c] for c in shap_imp.index], "média de |SHAP|": shap_imp.values.round(4),
                         "participação": (shap_imp / shap_imp.sum()).values.round(4)})
shap_regiao = pd.DataFrame()
if "regiao_residencia" in shap_var:
    shap_regiao = (shap_var["regiao_residencia"].groupby(teste["regiao_residencia"].values).agg(["mean", "size"])
                   .sort_values("mean", ascending=False).reset_index())
    shap_regiao.columns = ["região de residência", "SHAP médio (log-odds)", "agendamentos no teste"]
    shap_regiao = shap_regiao.round(4)
print("[11B] SHAP por variável:"); print(shap_tab.to_string(index=False))

# %% ETAPA 12 – PERFIS E PREVISÃO DE UM AGENDAMENTO NOVO --------------------------------
dados["chance prevista"] = pipe_r.predict_proba(dados[cols_r])[:, 1]
perfis = (dados.groupby(["sexo", "regiao_residencia", "faixa_etaria", "classe"])
          .agg(pessoas=("compareceu", "size"), **{"chance prevista": ("chance prevista", "mean"), "taxa real": ("compareceu", "mean"),
               "renda média (R$)": ("renda_media", "mean")}).reset_index())
perfis = perfis[(perfis["pessoas"] >= MIN_PERFIL) & (perfis["regiao_residencia"] != NA) &
                (perfis["classe"] != NA)].sort_values("chance prevista", ascending=False)
perfis["perfil"] = (perfis["sexo"].str.replace("Feminino", "Mulher").str.replace("Masculino", "Homem") + ", " +
                    perfis["regiao_residencia"].str.replace("Zona", "zona") + ", " + perfis["faixa_etaria"] +
                    ", classe " + perfis["classe"])
perfis["chance"] = np.where(perfis["chance prevista"] >= 0.5, "alta",
                            np.where(perfis["chance prevista"] <= 0.2, "baixa", "média"))
perfis = perfis[["perfil", "pessoas", "chance prevista", "taxa real", "renda média (R$)", "chance"]].round(3)
perfis["renda média (R$)"] = perfis["renda média (R$)"].round(0)

ref_bairro = dados.groupby("bairro")[["regiao_residencia", "classe", "renda_media"]].first()
ref_bairro["bairro"] = ref_bairro.index
ref_bairro.index = ref_bairro.index.map(norm)
def prever(sexo, raca_cor, faixa_etaria, bairro, regiao_atendimento):
    """Chance de comparecer de um agendamento novo, usando só campos da base."""
    linha = {"sexo": sexo, "raca_cor": raca_cor, "faixa_etaria": faixa_etaria, "bairro": bairro,
             "regiao_atendimento": regiao_atendimento, "regiao_residencia": NA, "classe": NA,
             "renda_media": np.nan, "ap": "Sem informação", "profissional": "Sem informação"}
    if norm(bairro) in ref_bairro.index:          # ignora maiúsculas e acentos
        linha.update(ref_bairro.loc[norm(bairro)].to_dict())
    return float(pipe_r.predict_proba(pd.DataFrame([linha])[cols_r])[:, 1][0])
exemplos = pd.DataFrame([
    ("Mulher, parda, 59 anos ou mais, mora na Maré, atendida na Zona Norte", prever("Feminino", "Parda", "59 anos ou mais", "Maré", "Zona norte")),
    ("Mulher, parda, 24 a 28 anos, mora na Maré, atendida na Zona Norte", prever("Feminino", "Parda", "24 a 28 anos", "Maré", "Zona norte")),
    ("Homem, branco, 59 anos ou mais, mora no Andaraí, atendido na Zona Norte", prever("Masculino", "Branca", "59 anos ou mais", "Andaraí", "Zona norte")),
    ("Mulher, preta, 29 a 33 anos, mora em Santa Cruz, atendida na Zona Oeste", prever("Feminino", "Preta", "29 a 33 anos", "Santa Cruz", "Zona oeste")),
    ("Homem, pardo, 44 a 48 anos, mora em Campo Grande, atendido na Zona Oeste", prever("Masculino", "Parda", "44 a 48 anos", "Campo Grande", "Zona oeste")),
], columns=["agendamento novo", "chance prevista de comparecer"]).round(3)
print(exemplos.to_string(index=False))

# %% ETAPA 13 – GRÁFICOS -------------------------------------------------------------------
# G1: Q1 e Q2
fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
for ax, (v, rot) in zip(axes.flat[:5], Q1_VARS.items()):
    g = q1[(q1["variável"] == rot) & (q1["agendamentos"] >= 30)]
    if v != "faixa_etaria":
        g = g.sort_values("taxa", ascending=False)
    ax.barh(g["categoria"], g["taxa"], color=AZUL)
    for yi, (t, n) in enumerate(zip(g["taxa"], g["agendamentos"])):
        ax.text(t + 0.006, yi, f"{pct(t)} (n={n:,})".replace(",", "."), va="center", fontsize=7.5)
    ax.axvline(media, color=VERMELHO, ls="--", lw=1)
    ax.set_xlim(0, 0.8); ax.invert_yaxis(); ax.set_title(rot, fontsize=10); ax.tick_params(axis="y", labelsize=8)
ax = axes.flat[5]
g = q2c[q2c["classe"] != NA]
ax.bar(g["classe"], g["taxa"], color=LARANJA)
for xi, (t, n) in enumerate(zip(g["taxa"], g["agendamentos"])):
    ax.text(xi, t + 0.008, f"{pct(t)}\n(n={int(n):,})".replace(",", "."), ha="center", fontsize=7.5)
ax.axhline(media, color=VERMELHO, ls="--", lw=1); ax.set_ylim(0, 0.6)
ax.set_title("Classe ABEP do bairro (C2 = mais pobre)", fontsize=10)
fig.suptitle(f"Taxa de comparecimento por grupo (linha vermelha = média geral, {pct(media)})", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{PASTA_SAIDA}/G1_perfil_comparecimento.png", dpi=150); plt.close()

# G2: modelos por cenário
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
ordem = list(CENARIOS)
for ax, mod in zip(axes, MODELOS_A_RODAR):
    t = tab[(tab["modelo"] == NOME[mod]) & (tab["balanceado"] == "sim")].set_index("cenário")
    x = np.arange(len(ordem))
    ax.plot(x, t.loc[ordem, "AUC treino"], "--o", color=CINZA, label="Treino")
    ax.plot(x, t.loc[ordem, "AUC teste"], "-s", color=COR[mod], lw=2.3, label="Teste")
    for xi, v in zip(x, t.loc[ordem, "AUC teste"]):
        ax.annotate(br(v), (xi, v), textcoords="offset points", xytext=(0, -15), ha="center", fontsize=8)
    ax.axhline(0.5, color=VERMELHO, ls=":", lw=1)
    ax.set_xticks(x, ordem)
    ax.set_title(NOME[mod], fontsize=11); ax.set_ylim(0.4, 1.0); ax.legend(fontsize=7, frameon=False, loc="upper left")
axes[0].set_ylabel("AUC")
fig.suptitle("Os 3 modelos em cada cenário (com balanceamento): AUC no treino e no teste", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{PASTA_SAIDA}/G2_modelos_por_cenario.png", dpi=150); plt.close()

# G3: importância por variável (permutação), 3 modelos
ordem_v = perm.groupby("variável")["queda na AUC ao embaralhar"].mean().sort_values().index
fig, ax = plt.subplots(figsize=(10, 4.8))
yb = np.arange(len(ordem_v)); h = 0.26
for k, mod in enumerate(MODELOS_A_RODAR):
    s = perm[perm["modelo"] == NOME[mod]].set_index("variável").loc[ordem_v, "queda na AUC ao embaralhar"]
    ax.barh(yb + (k - 1) * h, s.values, h, color=COR[mod], label=NOME[mod])
ax.set_yticks(yb, ordem_v); ax.set_xlabel("Queda na AUC quando a variável é embaralhada (quanto maior, mais importante)")
ax.legend(fontsize=8, frameon=False, loc="lower right")
ax.set_title(f"O que mais pesa no comparecimento – cenário {cen_r}", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{PASTA_SAIDA}/G3_importancia_por_variavel.png", dpi=150); plt.close()

# G4: validação
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1.3, 1, 1]})
ax = axes[0]
v = validacao.dropna(subset=["AUC"]).iloc[::-1]
cores = [VERDE if "recomendado" in s else (VERMELHO if ("embaralhados" in s or "Chute" in s) else CINZA) for s in v["teste"]]
ax.barh(v["teste"], v["AUC"], color=cores)
for yi, a in enumerate(v["AUC"]):
    ax.text(a + 0.005, yi, br(a), va="center", fontsize=8)
ax.axvline(0.5, color=VERMELHO, ls=":", lw=1); ax.set_xlim(0.3, 1.0); ax.tick_params(axis="y", labelsize=8)
ax.set_title("(a) Testes de validação (AUC)", fontsize=10)
ax = axes[1]
ax.plot(cv["janela"], cv["AUC"], "-o", color=AZUL, label="AUC")
ax.plot(cv["janela"], cv["taxa real de comparecimento"], "--s", color=CINZA, label="Taxa real de comparecimento")
ax.set_xticks(cv["janela"], [s.replace(" a ", "\na\n") for s in cv["período de teste"]], fontsize=7)
ax.set_ylim(0, 1); ax.legend(fontsize=8, frameon=False); ax.set_title("(b) Validação cruzada no tempo", fontsize=10)
ax = axes[2]
ax.plot([0, 1], [0, 1], ":", color=CINZA, label="Calibração perfeita")
ax.plot(calib["probabilidade média prevista"], calib["taxa real"], "-o", color=VERDE, label="Modelo")
ax.set_xlabel("Chance prevista"); ax.set_ylabel("Taxa real"); ax.legend(fontsize=8, frameon=False)
ax.set_title("(c) Calibração: previsto x real (10 faixas)", fontsize=10)
fig.suptitle(f"O resultado é real? Validação do modelo recomendado ({NOME[mod_v]}, {cen_r})", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{PASTA_SAIDA}/G4_validacao.png", dpi=150); plt.close()

# G5: perfis
top = pd.concat([perfis.head(8), perfis.tail(8)])
fig, ax = plt.subplots(figsize=(11, 6.6))
yy = np.arange(len(top))
ax.scatter(top["chance prevista"], yy, s=60, zorder=3, label="Chance prevista",
           color=[VERDE if c == "alta" else (VERMELHO if c == "baixa" else CINZA) for c in top["chance"]])
ax.scatter(top["taxa real"], yy, marker="|", s=140, color="black", zorder=3, label="Taxa real")
ax.axvline(media, color=CINZA, ls="--", lw=1); ax.axhline(7.5, color=CINZA, lw=0.6)
ax.set_yticks(yy, [f"{p} (n={n})" for p, n in zip(top["perfil"], top["pessoas"])], fontsize=8); ax.invert_yaxis()
ax.set_xlabel("Chance de comparecer"); ax.set_xlim(0, 1); ax.legend(fontsize=8, frameon=False, loc="lower right")
ax.set_title("Perfis com maior (topo) e menor (base) chance de comparecer", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{PASTA_SAIDA}/G5_perfis.png", dpi=150); plt.close()

# G6: SHAP do modelo recomendado
fig, axes = plt.subplots(1, 2 if len(shap_regiao) else 1, figsize=(12, 4.2), squeeze=False)
s6 = shap_tab.iloc[::-1]
ax = axes[0, 0]
ax.barh(s6["variável"], s6["participação"], color="#2F5496")
for i, v in enumerate(s6["participação"]):
    ax.text(v + 0.005, i, pct(v), va="center", fontsize=9)
ax.set_xlim(0, s6["participação"].max() * 1.2); ax.set_xlabel("Participação na soma das médias de |SHAP|")
ax.set_title("(a) Contribuição de cada variável", fontsize=11)
if len(shap_regiao):
    ax = axes[0, 1]
    r6 = shap_regiao.iloc[::-1]
    ax.barh(r6["região de residência"], r6["SHAP médio (log-odds)"],
            color=[VERDE if v > 0 else VERMELHO for v in r6["SHAP médio (log-odds)"]])
    ax.axvline(0, color="black", lw=0.8); ax.set_xlabel("SHAP médio da região (log-odds de comparecer)")
    ax.set_title("(b) Direção do efeito da região de residência", fontsize=11)
fig.suptitle(f"Valores SHAP do modelo recomendado ({NOME[mod_v]}, {cen_r}), conjunto de teste", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{PASTA_SAIDA}/G6_shap.png", dpi=160); plt.close()

# %% ETAPA 13B – PÓS-PROCESSAMENTO: DA PREVISÃO À AÇÃO -------------------------------------------
# Transforma a saída do modelo recomendado em informação de uso para a Clínica da Família:
#   chance de comparecer, faixa de prioridade, ação sugerida, motivo e resumo da agenda.
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
import joblib

# 13B.1 Calibração: avaliamos se a nota do modelo pode ser lida como chance real.
# Testamos reservar os 20% mais recentes do treino para calibrar (isotônica e Platt). Como a taxa de
# comparecimento muda no tempo, perder esses meses custa mais do que a calibração devolve; por isso, a
# chance usada é a do modelo completo, e a tabela abaixo documenta a comparação.
n_fit = int(len(treino) * 0.8)
tr_fit, tr_cal = treino.iloc[:n_fit], treino.iloc[n_fit:]
mod_80 = montar(cols_r, mod_v, True, tr_fit["compareceu"]).fit(tr_fit[cols_r], tr_fit["compareceu"])
s_cal, s_te = mod_80.predict_proba(tr_cal[cols_r])[:, 1], mod_80.predict_proba(teste[cols_r])[:, 1]
p_iso = IsotonicRegression(out_of_bounds="clip").fit(s_cal, tr_cal["compareceu"]).predict(s_te)
logit = lambda v: np.log(np.clip(v, 1e-6, 1 - 1e-6) / (1 - np.clip(v, 1e-6, 1 - 1e-6))).reshape(-1, 1)
p_platt = LogisticRegression().fit(logit(s_cal), tr_cal["compareceu"]).predict_proba(logit(s_te))[:, 1]
calib_comp = pd.DataFrame([
    {"versão": "Modelo completo (usado)", "AUC": roc_auc_score(y_te, prob), "Brier": brier_score_loss(y_te, prob),
     "chance média prevista": prob.mean(), "taxa real": y_te.mean()},
    {"versão": "80% do treino + calibração isotônica", "AUC": roc_auc_score(y_te, p_iso), "Brier": brier_score_loss(y_te, p_iso),
     "chance média prevista": p_iso.mean(), "taxa real": y_te.mean()},
    {"versão": "80% do treino + calibração de Platt", "AUC": roc_auc_score(y_te, p_platt), "Brier": brier_score_loss(y_te, p_platt),
     "chance média prevista": p_platt.mean(), "taxa real": y_te.mean()},
    {"versão": "Referência: prever a taxa média para todos", "AUC": 0.5, "Brier": brier_score_loss(y_te, np.full(len(y_te), y_tr.mean())),
     "chance média prevista": y_tr.mean(), "taxa real": y_te.mean()}]).round(4)
base_prod, iso = pipe_r, None

def chance_comparecer(df):
    """p = f(x): chance de comparecer dada pelo modelo recomendado (já próxima da taxa real; ver calib_comp)."""
    return base_prod.predict_proba(df[cols_r])[:, 1]

p_cal = chance_comparecer(teste)
cc = pd.DataFrame({"p": p_cal, "y": y_te.values})
cc["faixa"] = pd.qcut(cc["p"].rank(method="first"), 10, labels=False) + 1
calib_depois = cc.groupby("faixa").agg(**{"chance prevista média": ("p", "mean"), "taxa real": ("y", "mean"),
                                          "agendamentos": ("y", "size")}).reset_index().round(3)
print("[13B] Calibração:"); print(calib_comp.to_string(index=False))

# 13B.2 Motivo da chance: as duas variáveis com maior valor SHAP absoluto naquele agendamento
modelo_base = base_prod.named_steps["modelo"]
expl_prod = (shap.LinearExplainer(modelo_base, base_prod.named_steps["prep"].transform(treino[cols_r]))
             if mod_v == "RegressaoLogistica" else shap.TreeExplainer(modelo_base))
var_prod = [next(c for c in ordem_cols if n.split("__", 1)[1].startswith(c))
            for n in base_prod.named_steps["prep"].get_feature_names_out()]

def motivos(df, k=2):
    sv_ = expl_prod.shap_values(base_prod.named_steps["prep"].transform(df[cols_r]))
    sv_ = sv_[1] if isinstance(sv_, list) else (sv_[:, :, 1] if np.ndim(sv_) == 3 else sv_)
    agg_ = pd.DataFrame(sv_, columns=var_prod).T.groupby(level=0).sum().T
    textos = []
    for i in range(len(df)):
        linha = agg_.iloc[i]
        partes = []
        for c in linha.abs().sort_values(ascending=False).index[:k]:
            v = df.iloc[i][c]
            valor = f"R$ {v:,.0f}".replace(",", ".") if c == "renda_media" and pd.notna(v) else str(v)
            partes.append(f"{ROT[c]} = {valor} ({'aumenta' if linha[c] > 0 else 'reduz'} a chance)")
        textos.append("; ".join(partes))
    return textos

# 13B.3 Faixas, ações e resumo da agenda
LIMIARES = {"baixa": 0.20, "alta": 0.50}
ACOES = {"Baixa chance": "WhatsApp/SMS e e-mail 72 h e 24 h antes pedindo confirmação; sem resposta, ligação do ACS "
                         "oferecendo remarcação ou teleconsulta assistida na unidade",
         "Média chance": "WhatsApp/SMS 24 h antes pedindo confirmação, com o link também por e-mail",
         "Alta chance": "Lembrete padrão por WhatsApp/SMS com o link"}

def faixa_de(p):
    return np.where(p < LIMIARES["baixa"], "Baixa chance", np.where(p < LIMIARES["alta"], "Média chance", "Alta chance"))

def pos_processar(df):
    """Recebe agendamentos e devolve a lista de prioridade: chance, faixa, ação e motivo."""
    out = df.copy()
    out["chance de comparecer"] = chance_comparecer(df)
    out["faixa"] = faixa_de(out["chance de comparecer"])
    out["ação sugerida"] = out["faixa"].map(ACOES)
    out["motivo principal"] = motivos(df)
    out["prioridade"] = out["chance de comparecer"].rank(method="first").astype(int)
    return out.sort_values("prioridade")

def resumo_agenda(lista, capacidade=None):
    """Fórmulas do resumo: comparecimentos esperados = Σ p; ausências esperadas = Σ (1 − p);
    com k contatos nos de menor p: ausências alcançadas = Σ_k (1 − p); precisão = ausências alcançadas ÷ k;
    ganho sobre sorteio = precisão ÷ (ausências esperadas ÷ n)."""
    p = lista["chance de comparecer"].astype(float); n = len(lista)
    k = min(n, capacidade if capacidade is not None else int((lista["faixa"] == "Baixa chance").sum()))
    sel = lista.nsmallest(k, "chance de comparecer")
    aus_tot, aus_k = (1 - p).sum(), (1 - sel["chance de comparecer"]).sum()
    return {"agendamentos (n)": n, "comparecimentos esperados (Σp)": round(p.sum(), 1),
            "ausências esperadas (Σ(1−p))": round(aus_tot, 1), "contatos (k)": k,
            "ausências esperadas entre os contatados": round(float(aus_k), 1),
            "precisão esperada dos contatos": round(aus_k / k, 3) if k else None,
            "% das ausências alcançadas": round(aus_k / aus_tot, 3) if aus_tot else None,
            "ganho sobre sorteio": round((aus_k / k) / (aus_tot / n), 2) if k and aus_tot else None}

# 13B.4 Avaliação no teste
pp = teste.copy()
pp["chance"] = p_cal
pp["faixa"] = faixa_de(p_cal)
ORD_FX = ["Baixa chance", "Média chance", "Alta chance"]
pos_faixas = pp.groupby("faixa").agg(agendamentos=("compareceu", "size"), compareceram=("compareceu", "sum"),
                                     **{"chance média prevista": ("chance", "mean")}).reindex(ORD_FX).reset_index()
pos_faixas["% dos agendamentos"] = pos_faixas["agendamentos"] / len(pp)
pos_faixas["taxa real de comparecimento"] = pos_faixas["compareceram"] / pos_faixas["agendamentos"]
pos_faixas["ação sugerida"] = pos_faixas["faixa"].map(ACOES)
pos_faixas = pos_faixas.round(3)
ordem_pp = pp.sort_values("chance", kind="stable")
total_aus = (1 - pp["compareceu"]).sum()
pos_ganho = []
for f in [0.1, 0.2, 0.3, 0.5]:
    sel = ordem_pp.iloc[:int(len(ordem_pp) * f)]
    pos_ganho.append({"agendamentos contatados (os de menor chance)": f, "quantidade": len(sel),
                      "% deles que de fato não compareceu": round(1 - sel["compareceu"].mean(), 3),
                      "% previsto pelo modelo": round(1 - sel["chance"].mean(), 3),
                      "% de todas as ausências alcançadas": round((1 - sel["compareceu"]).sum() / total_aus, 3),
                      "% de ausências alcançadas por sorteio": f})
pos_ganho = pd.DataFrame(pos_ganho)
pos_regiao = []
for reg, g in pp.groupby("regiao_residencia"):
    auc_dentro = roc_auc_score(g["compareceu"], g["chance"]) if (len(g) >= 100 and g["compareceu"].nunique() == 2) else np.nan
    pos_regiao.append({"região de residência": reg, "agendamentos no teste": len(g),
                       "comparecimentos esperados (Σp)": round(g["chance"].sum(), 0), "comparecimentos reais": int(g["compareceu"].sum()),
                       "chance prevista média": round(g["chance"].mean(), 3), "taxa real": round(g["compareceu"].mean(), 3),
                       "AUC dentro da região": None if np.isnan(auc_dentro) else round(auc_dentro, 3)})
pos_regiao = pd.DataFrame(pos_regiao).sort_values("taxa real", ascending=False)

# 13B.5 Exemplo: a agenda do dia com mais agendamentos no teste
dia_demo = teste["dthr_marcacao"].dt.date.value_counts().idxmax()
agenda_dia = teste[teste["dthr_marcacao"].dt.date == dia_demo]
lista_dia = pos_processar(agenda_dia)
resumo_dia = resumo_agenda(lista_dia)
resumo_dia["comparecimentos reais (conferência)"] = int(agenda_dia["compareceu"].sum())
demo = lista_dia[["prioridade", "sexo", "raca_cor", "faixa_etaria", "bairro", "regiao_residencia", "chance de comparecer",
                  "faixa", "ação sugerida", "motivo principal", "compareceu"]].rename(columns={
    "raca_cor": "raça/cor", "faixa_etaria": "faixa etária", "regiao_residencia": "região", "compareceu": "compareceu (real)"})
demo["chance de comparecer"] = demo["chance de comparecer"].round(3)
print(f"[13B] Agenda de {dia_demo:%d/%m/%Y}: {resumo_dia}")

# 13B.6 Fórmulas do pós-processamento
pos_formulas = pd.DataFrame([
    ["Nota do modelo", "s = f(x)", "Probabilidade dada pelo XGBoost para as variáveis x do agendamento (sexo, raça/cor, faixa etária, região, classe e renda do bairro)."],
    ["Chance de comparecer", "p = s", "A nota do modelo já fica próxima da taxa real (Brier de 0,080); a calibração extra piorou o resultado e não foi aplicada."],
    ["Faixa de prioridade", "Baixa se p < 0,20; Média se 0,20 ≤ p < 0,50; Alta se p ≥ 0,50", "Define a ação da equipe para cada agendamento."],
    ["Prioridade", "ordem crescente de p", "Quem tem menor chance é contatado primeiro."],
    ["Comparecimentos esperados", "E[C] = Σ p", "Soma das chances da agenda: quantas pessoas devem comparecer."],
    ["Ausências esperadas", "E[A] = Σ (1 − p)", "Quantas pessoas devem faltar ou cancelar."],
    ["Ausências alcançadas com k contatos", "A_k = Σ (1 − p) nos k agendamentos de menor p", "Quantos faltosos a equipe alcança ligando para k pessoas."],
    ["Precisão dos contatos", "A_k ÷ k", "Parte das ligações que vai, de fato, para quem faltaria."],
    ["Ganho sobre sorteio", "(A_k ÷ k) ÷ (E[A] ÷ n)", "Quantas vezes o modelo é melhor que escolher ao acaso quem contatar."],
    ["Consultas recuperadas (simulação)", "R = r × A_k", "r é a parte dos contatados que passa a comparecer após a ação; deve ser medida em piloto."],
], columns=["indicador", "fórmula", "o que significa"])

# 13B.7 Pacote do modelo para uso na rotina (lido por prever_agenda.py)
joblib.dump({"base": base_prod, "iso": iso, "colunas": cols_r, "var_prod": var_prod, "ref_bairro": ref_bairro,
             "limiares": LIMIARES, "acoes": ACOES, "rot": ROT, "modelo": NOME[mod_v], "conjunto": cen_r,
             "na": NA, "treinado_ate": f"{treino['dthr_marcacao'].max():%d/%m/%Y}"},
            f"{PASTA_SAIDA}/modelo_comparecimento.joblib")

# atualiza os exemplos com a chance do modelo recomendado
def prever(sexo, raca_cor, faixa_etaria, bairro, regiao_atendimento):
    """Chance de comparecer de um agendamento novo, usando só campos da base."""
    linha = {"sexo": sexo, "raca_cor": raca_cor, "faixa_etaria": faixa_etaria, "bairro": bairro,
             "regiao_atendimento": regiao_atendimento, "regiao_residencia": NA, "classe": NA, "renda_media": np.nan}
    if norm(bairro) in ref_bairro.index:
        linha.update(ref_bairro.loc[norm(bairro)].to_dict())
    return float(chance_comparecer(pd.DataFrame([linha]))[0])
exemplos["chance prevista de comparecer"] = [prever(*a) for a in [
    ("Feminino", "Parda", "59 anos ou mais", "Maré", "Zona norte"), ("Feminino", "Parda", "24 a 28 anos", "Maré", "Zona norte"),
    ("Masculino", "Branca", "59 anos ou mais", "Andaraí", "Zona norte"), ("Feminino", "Preta", "29 a 33 anos", "Santa Cruz", "Zona oeste"),
    ("Masculino", "Parda", "44 a 48 anos", "Campo Grande", "Zona oeste")]]
exemplos["chance prevista de comparecer"] = exemplos["chance prevista de comparecer"].round(3)

# %% ETAPA 14 – PLANILHA COM LEGENDAS ----------------------------------------------------------
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

LEG = {
 "fluxo_amostra": ("Fluxo da amostra",
   "Quantos agendamentos entram na base de modelagem e como se dividem entre Compareceu e Não compareceu.",
   "Confirma a leitura da base e a proporção do alvo. Nenhum filtro além do previsto foi aplicado.",
   "A primeira tabela mostra o total; a segunda, cada estado da base e a classe do alvo em que ele entra."),
 "detector_vazamento": ("Detector de vazamento",
   "Compara a proporção de campos vazios entre quem compareceu e quem não compareceu.",
   "Um campo que só existe depois da consulta 'entrega a resposta' ao modelo. Isso é vazamento e invalida o resultado.",
   "Diferença acima de 30 pontos percentuais = vazamento. Essas variáveis ficaram fora dos modelos."),
 "variaveis": ("Variáveis do estudo",
   "Todas as colunas da base final tratada e o uso de cada uma. Nenhuma variável nova foi criada.",
   "Para o grupo saber defender por que cada variável entrou ou saiu.",
   "'Uso' diz se a variável entrou no modelo, só na descrição ou foi descartada."),
 "Q1_comparecimento": ("Q1 – Perfil de quem comparece",
   "Taxa de comparecimento por sexo, raça/cor, faixa etária e região.",
   "Responde à Q1: qual o perfil sociodemográfico e territorial de quem comparece.",
   "Taxa = compareceram ÷ agendamentos (fórmula na coluna E). Compare cada taxa com a média geral."),
 "Q2_renda": ("Q2 – Renda e comparecimento",
   "Taxa de comparecimento por classe ABEP e por quintil de renda do bairro, e o índice de Erreygers.",
   "Responde à Q2: o comparecimento muda conforme a renda?",
   "Erreygers positivo = comparecimento concentrado nos bairros mais ricos; negativo = nos mais pobres; zero = sem gradiente."),
 "regiao_sem_informacao": ("Região de residência com erro #N/D",
   "As linhas em que a célula de região de residência mostra o erro #N/D: o PROCV não encontrou o bairro na tabela de-para.",
   "O código não corrige a planilha; as linhas são sinalizadas para o grupo incluir esses bairros no de-para.",
   "Cada linha é um agendamento. Nos gráficos e nos modelos elas aparecem como 'Bairro fora do de-para'."),
 "testes_estatisticos": ("Testes estatísticos (qui-quadrado)",
   "Teste de associação entre cada variável e o comparecimento.",
   "Mostra se as diferenças de taxa entre os grupos são maiores do que o acaso explicaria, e com que força.",
   "p-valor < 0,05 = associação estatisticamente significativa. V de Cramér: < 0,1 fraca, 0,1 a 0,3 moderada, ≥ 0,3 forte."),
 "cenarios": ("Cenários e justificativa",
   "Os blocos de variáveis testados, do mais pessoal ao mais territorial.",
   "Somar um bloco por vez mostra quanto cada tipo de informação ajuda a prever o comparecimento.",
   "Leia na ordem C1 → C4: cada conjunto soma um bloco de variáveis ao anterior."),
 "experimentos_treino_teste": ("Experimentos: treino x teste",
   "As 24 rodadas: 3 modelos × 4 conjuntos de variáveis (C1 a C4), com e sem balanceamento.",
   "Comparar treino e teste lado a lado mostra se o modelo aprendeu de verdade ou só decorou.",
   "AUC: 0,5 = sorteio, 1 = perfeito. Recall: % de quem compareceu que o modelo encontrou. Gap acima de 0,10 = overfitting. PR-AUC: área sob a curva de precisão e revocação; a linha de base é a proporção de quem compareceu."),
 "comparacao_modelos": ("Comparação final dos 3 modelos",
   "A melhor rodada elegível de cada modelo e a diferença entre C2 e C4.",
   "Indica o modelo vencedor e se vale a pena usar o cenário mais completo.",
   "Elegível = com balanceamento e gap de no máximo 0,10. Se C4 − C2 ≤ 0,01, recomendamos o C2, que é mais simples."),
 "validacao": ("Validação: o resultado é real?",
   "Testes para conferir se o modelo encontrou um padrão verdadeiro.",
   "Um bom número no teste pode ser sorte ou erro; estes testes descartam essas hipóteses.",
   "Embaralhado perto de 0,5 = sinal real. Validação no tempo estável = funciona em outros períodos."),
 "validacao_temporal": ("Validação cruzada no tempo",
   "O modelo treinado no passado e testado em 5 períodos seguintes.",
   "A taxa de comparecimento mudou muito ao longo do tempo; precisamos saber se o modelo resiste a isso.",
   "Cada linha é um período de teste. AUC parecida em todos = modelo estável."),
 "calibracao": ("Calibração",
   "As previsões divididas em 10 faixas, comparando a chance prevista com o que aconteceu.",
   "Mostra se o percentual previsto pode ser lido como probabilidade real.",
   "Se 'probabilidade média prevista' ≈ 'taxa real', o modelo está calibrado."),
 "importancia_por_variavel": ("Importância por variável (permutação)",
   "Quanto a AUC cai quando embaralhamos cada variável, nos 3 modelos.",
   "É a mesma medida para os 3 modelos, então dá para comparar qual variável pesa mais.",
   "Quanto maior a queda, mais o modelo depende daquela variável."),
 "importancia_detalhada": ("Importância detalhada por categoria",
   "Razão de chances na Regressão Logística e importância nativa no Random Forest e no XGBoost.",
   "Mostra quais categorias (ex.: Zona Norte, classe C2) puxam a chance para cima ou para baixo.",
   "Razão de chances acima de 1 = mais chance de COMPARECER; abaixo de 1 = menos chance."),
 "perfis": ("Perfis de alta e baixa chance",
   "Combinações de sexo, região, idade e classe com pelo menos 30 pessoas.",
   "Percentuais isolados só segmentam; o perfil combinado mostra quem tem mais e menos chance de comparecer.",
   "Chance alta ≥ 50%, baixa ≤ 20%. Compare 'chance prevista' com 'taxa real'."),
 "exemplos_previsao": ("Exemplos de previsão",
   "A chance prevista para agendamentos novos, com a função prever() do script.",
   "Mostra como o modelo seria usado na prática, por exemplo para priorizar ligações de confirmação.",
   "Use a ordem (quem tem mais e menos chance), não o percentual exato."),
}
VARIAVEIS = pd.DataFrame([
    ["ID", "Criada pelo grupo", "Descartada", "Só identifica a linha."],
    ["area_programatica", "SMS-RJ", "Não usada", "O grupo preferiu apresentar o território por região."],
    ["Região de atendimento", "Criada pelo grupo", "Modelo (C3)", "Região da unidade, mais fácil de interpretar que a AP."],
    ["sexo", "SMS-RJ", "Modelo (C1)", "Fator predisponente; responde à Q1."],
    ["raca_cor", "SMS-RJ", "Modelo (C1)", "Fator predisponente e de equidade; responde à Q1."],
    ["bairro_residencia", "SMS-RJ", "Modelo (C4)", "Detalha o território."],
    ["Região (residência)", "Criada pelo grupo", "Modelo (C2)", "Território de moradia, divisão oficial da Prefeitura; usada como está na base."],
    ["Renda média por família", "Criada pelo grupo", "Modelo (C2)", "Condição socioeconômica do bairro; responde à Q2."],
    ["Classe associada", "Criada pelo grupo", "Modelo (C2)", "Leitura da renda em classes ABEP."],
    ["Faixa salário mínimo", "Criada pelo grupo", "Só descrição", "Repete a classe."],
    ["profissional_cbo_descricao", "SMS-RJ", "Não usada", "Não sabemos se registra quem foi agendado ou quem atendeu."],
    ["tipo_consulta", "SMS-RJ", "Descartada", "Vazamento: vazio em 98% das faltas."],
    ["dthr_marcacao", "SMS-RJ", "Divisão treino/teste", "Ordena os agendamentos no tempo."],
    ["dthr_inicio/fim_atendimento", "SMS-RJ", "Descartada", "Vazamento: só existe se houve consulta."],
    ["Mês/Ano da consulta", "Criada pelo grupo", "Descartada", "Vazamento (n/a nas faltas) e horários diferem por unidade."],
    ["estado_marcacao", "SMS-RJ", "Alvo", "Executado = Compareceu; Faltou e cancelamentos (paciente e profissional) = Não compareceu."],
    ["eh_coleta", "SMS-RJ", "Descartada", "Vazamento: vazio em 100% das faltas."],
    ["condicoes_ativas / Nome da doença", "SMS-RJ / grupo", "Descartada", "Vazamento: morbidade é registrada durante ou depois da consulta."],
    ["mes_ano_nascimento", "SMS-RJ", "Não usada", "A faixa etária do grupo já representa a idade; todos os registros mantidos."],
    ["Faixa etária", "Criada pelo grupo", "Modelo (C1)", "Fator predisponente; responde à Q1."],
], columns=["coluna da base", "origem", "uso", "por quê"])

ESCURO, CLARO = "1F3864", "D9E2F3"
fina = Side(style="thin", color="BFBFBF")
def aba(wb, nome, df, fmt=None, formula_taxa=False):
    ws = wb.create_sheet(nome)
    tit, oque, porque, ler = LEG[nome]
    ncol = len(df.columns) + (1 if formula_taxa else 0)
    ncol = max(ncol, 4)
    ws["A1"] = tit
    ws["A1"].font = Font(name="Arial", size=14, bold=True, color=ESCURO)
    for i, (rot, txt) in enumerate([("O que é", oque), ("Por que analisamos", porque), ("Como ler", ler)], start=2):
        ws.cell(i, 1, rot).font = Font(name="Arial", size=10, bold=True, color=ESCURO)
        ws.cell(i, 2, txt).font = Font(name="Arial", size=10)
        ws.cell(i, 2).alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=i, start_column=2, end_row=i, end_column=ncol)
        ws.row_dimensions[i].height = 30
        for c in range(1, ncol + 1):
            ws.cell(i, c).fill = PatternFill("solid", fgColor="F2F2F2")
    h = 6
    cols = list(df.columns)
    if formula_taxa:
        cols = [c for c in cols if c != "taxa"] + ["taxa (fórmula)"]
    for j, c in enumerate(cols, 1):
        cell = ws.cell(h, j, c)
        cell.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor=ESCURO)
        cell.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
        cell.border = Border(top=fina, bottom=fina, left=fina, right=fina)
    ws.row_dimensions[h].height = 32
    vals = df[[c for c in df.columns if c != "taxa"]] if formula_taxa else df
    for i, row in enumerate(vals.itertuples(index=False), start=h + 1):
        for j, v in enumerate(row, 1):
            if isinstance(v, (np.floating, float)) and np.isnan(v):
                v = None
            elif isinstance(v, np.generic):
                v = v.item()
            cell = ws.cell(i, j, v)
            cell.font = Font(name="Arial", size=10)
            cell.border = Border(top=fina, bottom=fina, left=fina, right=fina)
            cell.alignment = Alignment(wrap_text=isinstance(v, str) and len(v) > 40, vertical="top")
            if fmt and cols[j - 1] in fmt:
                cell.number_format = fmt[cols[j - 1]]
        if formula_taxa:
            ci, cc = cols.index("agendamentos") + 1, cols.index("compareceram") + 1
            cell = ws.cell(i, len(cols), f"={get_column_letter(cc)}{i}/{get_column_letter(ci)}{i}")
            cell.number_format = "0.0%"; cell.font = Font(name="Arial", size=10)
            cell.border = Border(top=fina, bottom=fina, left=fina, right=fina)
        if i % 2 == 0:
            for j in range(1, len(cols) + 1):
                ws.cell(i, j).fill = PatternFill("solid", fgColor="F7F9FC")
    for j, c in enumerate(cols, 1):
        tam = max([len(str(c))] + [len(str(x)) for x in vals.iloc[:, min(j - 1, vals.shape[1] - 1)].astype(str)][:200])
        ws.column_dimensions[get_column_letter(j)].width = min(max(12, tam * 0.95), 60)
    ws.column_dimensions["A"].width = max(ws.column_dimensions["A"].width, 22)
    ws.freeze_panes = ws.cell(h + 1, 1)
    ws.sheet_view.showGridLines = False
    return ws

LEG["shap"] = ("Valores SHAP do modelo recomendado",
   "Contribuição de cada variável para as previsões do modelo recomendado, calculada no conjunto de teste.",
   "SHAP é a técnica padrão para explicar previsões de modelos de árvores (Lundberg e Lee, 2017) e complementa a importância por permutação.",
   "Participação = parte da soma das médias de |SHAP|. Na segunda tabela, SHAP médio positivo = a região aumenta a chance de comparecer.")
LEG["pos_formulas"] = ("Pós-processamento: fórmulas",
   "Todas as fórmulas usadas para transformar a previsão em chance, faixa, prioridade e resumo da agenda.",
   "Deixa o cálculo transparente e reproduzível para quem vai usar ou avaliar a ferramenta.",
   "p é a chance de comparecer de cada agendamento; n é o total de agendamentos; k é o número de contatos.")
LEG["pos_calibracao"] = ("Pós-processamento: calibração",
   "Teste de calibração: o modelo completo comparado com versões calibradas (isotônica e Platt), e a chance prevista por faixa no teste.",
   "Para dizer a uma pessoa qual é a sua chance, o número precisa ser próximo da taxa real; o teste mostra qual versão chega mais perto.",
   "Brier menor = chances mais próximas do que aconteceu. Na segunda tabela, compare a chance prevista média com a taxa real.")
LEG["pos_agenda_exemplo"] = ("Pós-processamento: exemplo de lista de prioridade",
   "A lista que a Clínica da Família receberia para a agenda de um dia real do período de teste.",
   "Mostra o produto final: chance de cada pessoa, faixa, ação sugerida e motivo, na ordem de contato.",
   "A coluna 'compareceu (real)' só existe aqui para conferência; na rotina, ela é desconhecida.")
LEG["pos_faixas"] = ("Pós-processamento: faixas de prioridade",
   "Os agendamentos do teste agrupados pela chance de comparecer, com a taxa real e a ação sugerida para cada faixa.",
   "Transforma a previsão em uma regra de uso para a Clínica da Família: quem recebe confirmação ativa, lembrete ou rotina padrão.",
   "Compare a taxa real entre as faixas: quanto mais separadas, mais útil é a priorização.")
LEG["pos_ganho"] = ("Pós-processamento: ganho ao priorizar contatos",
   "Quantas ausências seriam alcançadas contatando primeiro os agendamentos de menor chance de comparecer.",
   "Mostra o ganho de usar o modelo em vez de escolher agendamentos ao acaso, com a mesma capacidade de contato da equipe.",
   "Ex.: contatar 30% dos agendamentos alcança X% das ausências; por sorteio, alcançaria 30%.")
LEG["pos_regiao"] = ("Pós-processamento: leitura por território",
   "Comparecimentos esperados (Σp) e reais, chance média, taxa real e AUC dentro de cada região, no teste.",
   "Indica o que a gestão municipal pode acompanhar por território e até onde o modelo distingue pessoas da mesma região.",
   "AUC dentro da região perto de 0,5 = o modelo quase não diferencia pessoas da mesma região; a decisão é territorial.")
if len(sem_regiao) == 0:          # sem linhas a sinalizar: a aba não é criada
    LEG.pop("regiao_sem_informacao")
wb = Workbook()
leia = wb.active; leia.title = "LEIA-ME"
leia["A1"] = "Predição de comparecimento às teleconsultas do SUS-RJ – resultados (versão 3)"
leia["A1"].font = Font(name="Arial", size=14, bold=True, color=ESCURO)
leia["A2"] = (f"Base: aba 'Base final tratada' da planilha v3. Base de modelagem: {len(dados):,} agendamentos "
              f"({pct(media)} compareceram). Alvo: Compareceu x Não compareceu (faltas + cancelamentos do paciente e do profissional). "
              f"Território por região. Valores calculados pelo script estudo_comparecimento_final.py; "
              f"as taxas das abas Q1 e Q2 são fórmulas.").replace(",", ".", 1)
leia["A2"].font = Font(name="Arial", size=10); leia["A2"].alignment = Alignment(wrap_text=True)
leia.merge_cells("A2:C2"); leia.row_dimensions[2].height = 45
for j, c in enumerate(["aba", "o que mostra", "pergunta ou etapa"], 1):
    cell = leia.cell(4, j, c); cell.font = Font(name="Arial", bold=True, color="FFFFFF")
    cell.fill = PatternFill("solid", fgColor=ESCURO)
ETAPA = {"Q1_comparecimento": "Q1", "Q2_renda": "Q2", "importancia_por_variavel": "Q3",
         "importancia_detalhada": "Q3", "perfis": "Q3 e resultado final", "exemplos_previsao": "Uso prático",
         "testes_estatisticos": "Q1 e Q2", "regiao_sem_informacao": "Qualidade da base", "shap": "Q3",
         "pos_faixas": "Uso do modelo", "pos_ganho": "Uso do modelo", "pos_regiao": "Uso do modelo",
         "pos_formulas": "Uso do modelo", "pos_calibracao": "Uso do modelo", "pos_agenda_exemplo": "Uso do modelo"}
for i, (k, (t, o, _, _)) in enumerate(LEG.items(), start=5):
    for j, v in enumerate([k, o, ETAPA.get(k, "Método")], 1):
        leia.cell(i, j, v).font = Font(name="Arial", size=10)
        leia.cell(i, j).alignment = Alignment(wrap_text=True, vertical="top")
leia.column_dimensions["A"].width = 28; leia.column_dimensions["B"].width = 80; leia.column_dimensions["C"].width = 22
leia.sheet_view.showGridLines = False

P3, P1, R0 = "0.000", "0.0%", "#,##0"
ws = aba(wb, "fluxo_amostra", pd.DataFrame(fluxo, columns=["etapa", "agendamentos"]), {"agendamentos": R0})
r0 = ws.max_row + 2
for j, c in enumerate(composicao.columns, 1):
    cell = ws.cell(r0, j, c); cell.font = Font(name="Arial", bold=True, color="FFFFFF"); cell.fill = PatternFill("solid", fgColor=ESCURO)
for i, r in enumerate(composicao.itertuples(index=False), start=r0 + 1):
    for j, v in enumerate(r, 1):
        cell = ws.cell(i, j, v.item() if isinstance(v, np.generic) else v); cell.font = Font(name="Arial", size=10)
    ws.cell(i, 4).number_format = P1
aba(wb, "detector_vazamento", vazamento, {"% vazio entre quem compareceu": P1, "% vazio entre quem não compareceu": P1})
aba(wb, "variaveis", VARIAVEIS)
aba(wb, "Q1_comparecimento", q1, {"agendamentos": R0, "compareceram": R0}, formula_taxa=True)
ws = aba(wb, "Q2_renda", q2c, {"agendamentos": R0, "compareceram": R0, "renda média do grupo (R$)": R0}, formula_taxa=True)
r0 = ws.max_row + 2
for j, c in enumerate(["quintil", "agendamentos", "compareceram", "renda média do quintil (R$)", "taxa (fórmula)"], 1):
    cell = ws.cell(r0, j, c); cell.font = Font(name="Arial", bold=True, color="FFFFFF"); cell.fill = PatternFill("solid", fgColor=ESCURO)
for i, r in enumerate(q2q.itertuples(index=False), start=r0 + 1):
    for j, v in enumerate([str(r[0]), int(r[1]), int(r[2]), float(r[4])], 1):
        ws.cell(i, j, v).font = Font(name="Arial", size=10)
    ws.cell(i, 5, f"=C{i}/B{i}").number_format = P1
r1 = ws.max_row + 2
for k, (rot, v) in enumerate([("Índice de Erreygers (base toda)", E), ("IC 95% – inferior", ic_E[0]), ("IC 95% – superior", ic_E[1]),
                              ("Erreygers só na Zona Norte", E_reg["Zona norte"]), ("Erreygers só na Zona Oeste", E_reg["Zona oeste"])]):
    ws.cell(r1 + k, 1, rot).font = Font(name="Arial", size=10, bold=True)
    ws.cell(r1 + k, 2, round(float(v), 4)).font = Font(name="Arial", size=10)
aba(wb, "testes_estatisticos", testes)
if len(sem_regiao):
    aba(wb, "regiao_sem_informacao", sem_regiao.rename(columns={"bairro": "bairro de residência", "estado_marcacao": "estado na base"}))
aba(wb, "cenarios", JUSTIFICATIVA)
aba(wb, "experimentos_treino_teste", tab.drop(columns="id"), {c: P3 for c in num_cols})
aba(wb, "comparacao_modelos", comp.drop(columns="id"))
aba(wb, "validacao", validacao, {"AUC": P3, "acurácia": P3})
aba(wb, "validacao_temporal", cv)
aba(wb, "calibracao", calib)
aba(wb, "importancia_por_variavel", perm.sort_values(["modelo", "queda na AUC ao embaralhar"], ascending=[True, False]))
aba(wb, "importancia_detalhada", detalhe)
aba(wb, "pos_formulas", pos_formulas)
ws = aba(wb, "pos_calibracao", calib_comp, {"chance média prevista": P1, "taxa real": P1})
r0 = ws.max_row + 2
for j, c in enumerate(calib_depois.columns, 1):
    cell = ws.cell(r0, j, c); cell.font = Font(name="Arial", bold=True, color="FFFFFF"); cell.fill = PatternFill("solid", fgColor=ESCURO)
for i, r in enumerate(calib_depois.itertuples(index=False), start=r0 + 1):
    for j, v in enumerate(r, 1):
        ws.cell(i, j, v.item() if isinstance(v, np.generic) else v).font = Font(name="Arial", size=10)
aba(wb, "pos_agenda_exemplo", demo, {"chance de comparecer": P1})
aba(wb, "pos_faixas", pos_faixas, {"% dos agendamentos": P1, "taxa real de comparecimento": P1, "chance média prevista": P1})
aba(wb, "pos_ganho", pos_ganho, {c: P1 for c in pos_ganho.columns if c != "quantidade"})
aba(wb, "pos_regiao", pos_regiao, {"chance prevista média": P1, "taxa real": P1})
ws = aba(wb, "shap", shap_tab, {"participação": P1})
if len(shap_regiao):
    r0 = ws.max_row + 2
    for j, c in enumerate(shap_regiao.columns, 1):
        cell = ws.cell(r0, j, c); cell.font = Font(name="Arial", bold=True, color="FFFFFF"); cell.fill = PatternFill("solid", fgColor=ESCURO)
    for i, r in enumerate(shap_regiao.itertuples(index=False), start=r0 + 1):
        for j, v in enumerate(r, 1):
            ws.cell(i, j, v.item() if isinstance(v, np.generic) else v).font = Font(name="Arial", size=10)
aba(wb, "perfis", perfis, {"chance prevista": P1, "taxa real": P1, "renda média (R$)": R0})
aba(wb, "exemplos_previsao", exemplos, {"chance prevista de comparecer": P1})
wb.save(f"{PASTA_SAIDA}/resultados_comparecimento_final.xlsx")

# Resultados também em JSON e pickle, para reuso em outras análises
res = {"n": len(dados), "media": media, "corte": f"{CORTE:%d/%m/%Y}", "n_tr": len(treino), "n_te": len(teste),
       "tx_tr": y_tr.mean(), "tx_te": y_te.mean(), "E": E, "ic_E": list(ic_E), "E_reg": E_reg,
       "vencedor": vencedor["id"], "recomendado": recomendado, "auc_vencedor": float(vencedor["AUC teste (melhor)"]),
       "auc_melhor_c2": float(melhor_c2["AUC teste"]), "auc_r": auc_r, "ic_auc": list(ic_auc),
       "fluxo": fluxo, "n_prof": n_prof, "n_sem_regiao": len(sem_regiao),
       "sem_regiao_bairros": sem_regiao["bairro"].value_counts().head(6).to_dict(),
       "n_comp": int(dados["compareceu"].sum()), "n_nao": int((1 - dados["compareceu"]).sum()),
       "pr_auc_r": float(t_id.loc[recomendado, "PR-AUC teste"]),
       "dia_demo": f"{dia_demo:%d/%m/%Y}", "resumo_dia": resumo_dia}
json.dump(res, open(f"{PASTA_SAIDA}/resumo.json", "w"), ensure_ascii=False, default=float, indent=1)
for nome, df in [("tab", tab), ("comp", comp), ("val", validacao), ("cv", cv), ("perm", perm), ("det", detalhe),
                 ("perfis", perfis), ("ex", exemplos), ("q1", q1), ("q2c", q2c), ("q2q", q2q), ("testes", testes), ("comp_alvo", composicao), ("shap_tab", shap_tab), ("shap_regiao", shap_regiao),
                 ("pos_faixas", pos_faixas), ("pos_ganho", pos_ganho), ("pos_regiao", pos_regiao),
                 ("calib_comp", calib_comp), ("calib_depois", calib_depois), ("demo", demo), ("pos_formulas", pos_formulas),
                 ("vaz", vazamento), ("calib", calib)]:
    df.to_pickle(f"{PASTA_SAIDA}/{nome}.pkl")
print(f"[14] Planilha e gráficos salvos em '{PASTA_SAIDA}/'")

# %% ETAPA 15 – RESULTADOS DO ESTUDO ---------------------------------------------------
L = "=" * 78
print(f"\n{L}\nRESULTADOS DO ESTUDO – PREDIÇÃO DE COMPARECIMENTO ÀS TELECONSULTAS DO SUS-RJ\n{L}")
mil = lambda v: f"{int(v):,}".replace(",", ".")
print(f"Base: {mil(len(dados))} agendamentos | Compareceu {mil(dados['compareceu'].sum())} ({pct(media)}) | "
      f"Não compareceu {mil((1 - dados['compareceu']).sum())} ({pct(1 - media)})")
print("\nQ1 – Taxa de comparecimento por região de residência:")
for _, r in q1[q1["variável"] == "Região de residência"].sort_values("taxa", ascending=False).iterrows():
    print(f"   {r['categoria']:<22} {pct(r['taxa']):>7}   ({int(r['compareceram'])} ÷ {int(r['agendamentos'])})")
print("\nQ1 – Associação com o comparecimento (qui-quadrado, V de Cramér):")
for _, r in testes.iterrows():
    print(f"   {r['variável']:<24} V = {br(r['V de Cramér'])}  ({r['força da associação']})")
print(f"\nQ2 – Índice de Erreygers (renda do bairro): {br(E, 4)} (IC 95%: {br(ic_E[0], 4)} a {br(ic_E[1], 4)})")
print(f"     Dentro da Zona Norte: {br(E_reg['Zona norte'], 4)} | dentro da Zona Oeste: {br(E_reg['Zona oeste'], 4)}")
print("\nQ3 – Melhor rodada de cada modelo (teste):")
for _, r in comp.iterrows():
    print(f"   {r['modelo']:<20} {r['melhor cenário']}  AUC {br(r['AUC teste (melhor)'])} | recall {pct(r['recall teste (melhor)'])} | "
          f"acurácia {pct(r['acurácia teste (melhor)'])} | AUC no C2 {br(r['AUC teste C2'])}")
rl = t_id.loc[recomendado]
print(f"\nModelo recomendado: {NOME[mod_v]} no {cen_r} – AUC {br(rl['AUC teste'])} (IC 95%: {br(ic_auc[0])} a {br(ic_auc[1])}), "
      f"acurácia {pct(rl['acurácia teste'])}, recall {pct(rl['recall teste'])}, PR-AUC {br(rl['PR-AUC teste'])} "
      f"(linha de base {br(y_te.mean())})")
print("\nValidação:")
for _, r in validacao.iterrows():
    print(f"   {r['teste']:<48} AUC {br(r['AUC'])}")
print("\nQ3 – Importância por variável (queda média na AUC, 3 modelos):")
for v, q in perm.groupby("variável")["queda na AUC ao embaralhar"].mean().sort_values(ascending=False).items():
    print(f"   {v:<24} {br(q)}")
print(f"\nQ3 – Valores SHAP do modelo recomendado ({NOME[mod_v]}, {cen_r}):")
for _, r in shap_tab.iterrows():
    print(f"   {r['variável']:<24} {pct(r['participação'])}")
print(f"\nCalibração: Brier do modelo usado {br(calib_comp.iloc[0]['Brier'])}; com calibração isotônica {br(calib_comp.iloc[1]['Brier'])}; "
      f"referência (taxa média) {br(calib_comp.iloc[3]['Brier'])}")
print("\nUso do modelo – faixas de prioridade (teste):")
for _, r in pos_faixas.iterrows():
    print(f"   {r['faixa']:<28} {pct(r['% dos agendamentos'])} dos agendamentos | comparecimento real {pct(r['taxa real de comparecimento'])}")
g30 = pos_ganho[pos_ganho.iloc[:, 0] == 0.3].iloc[0]
print(f"   Contatar os 30% de menor chance alcança {pct(g30['% de todas as ausências alcançadas'])} das ausências (sorteio: 30,0%)")
print("\nPerfis com MAIOR chance de comparecer:")
for _, r in perfis.head(5).iterrows():
    print(f"   {r['perfil']:<55} {pct(r['chance prevista'])} (real {pct(r['taxa real'])}, n={r['pessoas']})")
print("Perfis com MENOR chance de comparecer:")
for _, r in perfis.tail(5).iterrows():
    print(f"   {r['perfil']:<55} {pct(r['chance prevista'])} (real {pct(r['taxa real'])}, n={r['pessoas']})")
print(L)
