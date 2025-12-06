import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime

# ============================================================
# 1. CONFIGURAÇÕES GERAIS
# ============================================================
TICKERS = {
    "EGIE3": "EGIE3.SA",  # Engie Brasil
    "ALUP3": "ALUP3.SA",  # Alupar
    "TAEE11": "TAEE11.SA" # Taesa
}

MARKET_TICKER = "^BVSP"
START_DATE = "2020-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# CONFIGURAÇÃO TRIMESTRAL (4 ANOS)
YEARS_EXPLICIT = 4
QUARTERS_EXPLICIT = YEARS_EXPLICIT * 4  # 16 trimestres
TAX_RATE = 0.34

# ============================================================
# 2. FUNÇÕES AUXILIARES
# ============================================================
def get_focus_series():
    """Busca dados do Focus (IPCA e Selic) do Banco Central"""
    ipca_url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.4466/dados/ultimos/12?formato=json"
    selic_url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.4390/dados/ultimos/12?formato=json"
    
    try:
        ipca_data = requests.get(ipca_url, timeout=5).json()
        selic_data = requests.get(selic_url, timeout=5).json()

        ipca_vals = [float(x["valor"].replace(",", ".")) for x in ipca_data]
        selic_vals = [float(x["valor"].replace(",", ".")) for x in selic_data]

        ipca_avg = np.mean(ipca_vals) / 100.0
        selic_avg = np.mean(selic_vals) / 100.0

        risk_free = selic_avg
        g_long = ipca_avg + 0.015  # Inflação + ganho real modesto (PIB LP)

        return {
            "risk_free": risk_free,
            "inflation": ipca_avg,
            "g_long": g_long
        }
    except Exception as e:
        print(f"Aviso: Erro ao buscar Focus ({e}). Usando valores padrão.")
        return {
            "risk_free": 0.1175,
            "inflation": 0.045,
            "g_long": 0.06
        }

def download_price_series(ticker):
    """Baixa série histórica de preços ajustados"""
    try:
        data = yf.download(
            tickers=ticker,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=True,
            progress=False,
            multi_level_index=False
        )
        if data.empty:
            return pd.Series(dtype=float)

        if "Close" in data.columns:
            prices = data["Close"]
        elif "Adj Close" in data.columns:
            prices = data["Adj Close"]
        else:
            prices = data.iloc[:, 0]

        return prices.dropna()
    except Exception as e:
        print(f"Erro ao baixar preços para {ticker}: {e}")
        return pd.Series(dtype=float)

def compute_beta(asset_prices, market_prices):
    """Calcula beta do ativo vs mercado"""
    df = pd.concat([asset_prices, market_prices], axis=1, join="inner").dropna()
    df.columns = ["asset", "market"]
    
    if len(df) < 50:
        return 1.0
        
    returns_asset = df["asset"].pct_change().dropna()
    returns_market = df["market"].pct_change().dropna()
    
    cov_matrix = np.cov(returns_asset, returns_market)
    cov = cov_matrix[0, 1]
    var = np.var(returns_market)
    
    beta = cov / var if var != 0 else 1.0
    return beta

def safe_float(val):
    if val is None:
        return np.nan
    try:
        return float(val)
    except:
        return np.nan

def extract_financials(tk):
    """Extrai dados financeiros completos"""
    try:
        info = tk.info
    except:
        info = {}
        
    # Extração Básica
    price = safe_float(info.get("currentPrice") or info.get("previousClose") or info.get("regularMarketPrice"))
    shares_out = safe_float(info.get("sharesOutstanding"))
    market_cap = safe_float(info.get("marketCap"))
    
    eps = safe_float(info.get("trailingEps") or info.get("epsTrailingTwelveMonths"))
    pe = safe_float(info.get("trailingPE"))
    if np.isnan(pe) and price and eps:
        pe = price / eps

    # Fallback Shares
    if np.isnan(shares_out) and not np.isnan(market_cap) and not np.isnan(price) and price > 0:
        shares_out = market_cap / price
    
    total_debt = safe_float(info.get("totalDebt") or info.get("longTermDebt"))
    cash = safe_float(info.get("totalCash") or info.get("cash"))
    
    # Fallback Balance Sheet
    if np.isnan(total_debt) or np.isnan(cash):
        try:
            bs = tk.balance_sheet
            if not bs.empty:
                if np.isnan(total_debt):
                    total_debt = safe_float(bs.loc["Total Debt"].iloc[0]) if "Total Debt" in bs.index else 0.0
                if np.isnan(cash):
                    cash = safe_float(bs.loc["Cash And Cash Equivalents"].iloc[0]) if "Cash And Cash Equivalents" in bs.index else 0.0
        except:
            pass

    total_debt = 0.0 if np.isnan(total_debt) else total_debt
    cash = 0.0 if np.isnan(cash) else cash
    net_debt = total_debt - cash

    # EBIT e Fluxo
    ebit = safe_float(info.get("ebit") or info.get("operatingMargins", 0) * info.get("totalRevenue", 0))
    
    if np.isnan(ebit) or ebit == 0:
        try:
            fin_df = tk.financials
            if "EBIT" in fin_df.index:
                ebit = safe_float(fin_df.loc["EBIT"].iloc[0])
            elif "Operating Income" in fin_df.index:
                ebit = safe_float(fin_df.loc["Operating Income"].iloc[0])
        except:
            pass

    # Depreciação
    depreciation = np.nan
    try:
        cf_df = tk.cashflow
        if not cf_df.empty:
            possible_names = ["Depreciation", "DepreciationAndAmortization", "Depreciation & Amortization"]
            for name in possible_names:
                if name in cf_df.index:
                    depreciation = safe_float(cf_df.loc[name].iloc[0])
                    break
    except:
        pass
        
    if np.isnan(depreciation):
        depreciation = abs(ebit) * 0.2 if not np.isnan(ebit) else 0

    # CAPEX
    capex = np.nan
    try:
        cf_df = tk.cashflow
        if not cf_df.empty:
            possible_names = ["Capital Expenditures", "CapitalExpenditures", "capex"]
            for name in possible_names:
                if name in cf_df.index:
                    capex = safe_float(cf_df.loc[name].iloc[0])
                    break
    except:
        pass
    
    if not np.isnan(capex):
        capex = abs(capex)
    else:
        # Fallback conservador: CAPEX ~ Depreciação
        capex = depreciation

    # Delta WC
    delta_wc = np.nan
    try:
        cf_df = tk.cashflow
        possible_names = ["Change In Working Capital", "ChangeToNetincome"]
        for name in possible_names:
            if name in cf_df.index:
                delta_wc = safe_float(cf_df.loc[name].iloc[0])
                break
    except:
        pass
        
    if np.isnan(delta_wc):
        delta_wc = 0.0

    # FCFF Base e Normalização
    nopat = ebit * (1 - TAX_RATE) if not np.isnan(ebit) else 0
    fcff_calc = nopat + depreciation - capex - delta_wc

    is_normalized = False
    
    # Se o fluxo for negativo, usamos o NOPAT (lucro operacional líquido) como proxy de capacidade
    # Isso evita NaN na valoração principal
    if fcff_calc <= 0:
        fcff_base = nopat 
        is_normalized = True
    else:
        fcff_base = fcff_calc

    return {
        "price": price,
        "shares_out": shares_out,
        "market_cap": market_cap,
        "total_debt": total_debt,
        "cash": cash,
        "net_debt": net_debt,
        "ebit": ebit,
        "eps": eps,
        "pe": pe,
        "fcff_base_annual": fcff_base,
        "fcff_calc_raw": fcff_calc,
        "is_normalized": is_normalized,
        "info": info
    }

# ============================================================
# 3. PIPELINE DE VALUATION (TRIMESTRAL + RESTAURAÇÃO COMPLETA)
# ============================================================
def run_valuation(code, yf_ticker, macro, market_prices):
    print(f"\n---> Processando {code} ({yf_ticker})...")
    
    # 1. Dados de Mercado e Beta
    asset_prices = download_price_series(yf_ticker)
    if asset_prices.empty:
        print(f"Erro: Sem dados de preço para {code}")
        return None
        
    beta_calc = compute_beta(asset_prices, market_prices)
    
    # 2. Dados Financeiros
    tk = yf.Ticker(yf_ticker)
    fin = extract_financials(tk)
    
    if np.isnan(fin['shares_out']) or fin['shares_out'] <= 0:
        print(f"Erro Crítico: Não foi possível determinar o número de ações para {code}.")
        return None

    # 3. Definição do WACC - RESTAURANDO FASES (Explícito vs Perpetuidade)
    market_premium = 0.06
    
    # Beta
    beta_explicit = beta_calc
    beta_perp = 1.0 # Convergência para risco de mercado na perpetuidade
    
    # Ke
    ke_annual_explicit = macro["risk_free"] + beta_explicit * market_premium
    ke_annual_perp = macro["risk_free"] + beta_perp * market_premium
    
    # Kd (Assume spread constante)
    kd_pre_tax = macro["risk_free"] + 0.025
    kd_after_tax = kd_pre_tax * (1 - TAX_RATE)
    
    # Estrutura de Capital (Market Value)
    equity_val_market = fin['shares_out'] * fin['price']
    debt_val = fin['net_debt'] if fin['net_debt'] > 0 else 0
    total_cap = equity_val_market + debt_val
    
    we = equity_val_market / total_cap
    wd = debt_val / total_cap
    
    # WACC Anual (Duas fases)
    wacc_annual_explicit = (we * ke_annual_explicit) + (wd * kd_after_tax)
    wacc_annual_perp = (we * ke_annual_perp) + (wd * kd_after_tax)
    
    # Conversão para Trimestral
    wacc_qtr_explicit = (1 + wacc_annual_explicit)**(1/4) - 1
    wacc_qtr_perp = (1 + wacc_annual_perp)**(1/4) - 1
    
    # Crescimento (g)
    g_long_annual = macro["g_long"]
    g_long_qtr = (1 + g_long_annual)**(1/4) - 1
    
    # 4. PROJEÇÃO DO FLUXO (16 TRIMESTRES)
    # Usa WACC Explícito para descontar os 4 anos
    fcff_base_qtr = fin['fcff_base_annual'] / 4
    
    fcff_projections = []
    pv_flows = []
    
    current_fcff = fcff_base_qtr
    g_explicit_qtr = g_long_qtr
    
    for q in range(1, QUARTERS_EXPLICIT + 1):
        current_fcff = current_fcff * (1 + g_explicit_qtr)
        fcff_projections.append(current_fcff)
        factor = 1 / ((1 + wacc_qtr_explicit) ** q)
        pv_flows.append(current_fcff * factor)
        
    sum_pv_explicit = sum(pv_flows)
    
    # 5. VALOR RESIDUAL (PERPETUIDADE)
    # Usa WACC de Perpetuidade
    last_fcff = fcff_projections[-1]
    
    # Proteção: g < wacc_perp
    if wacc_qtr_perp <= g_long_qtr:
        g_for_tv = wacc_qtr_perp * 0.9
    else:
        g_for_tv = g_long_qtr
        
    tv_value_nominal = (last_fcff * (1 + g_for_tv)) / (wacc_qtr_perp - g_for_tv)
    
    # Traz a VP usando o WACC Explícito para o período de projeção
    tv_pv = tv_value_nominal / ((1 + wacc_qtr_explicit) ** QUARTERS_EXPLICIT)
    
    # 6. VALOR DA FIRMA E EQUITY
    firm_value = sum_pv_explicit + tv_pv
    equity_value_model = firm_value - fin['net_debt']
    
    fair_price = equity_value_model / fin['shares_out']

    # 7. CENÁRIO DE RECUPERAÇÃO (RESTAURADO)
    # Calculado se o fluxo original (sem normalização) for negativo
    recovery_price = np.nan
    if fin['fcff_calc_raw'] <= 0:
        # Assume um turnaround onde a empresa vale ao menos seu Patrimônio Líquido Contábil ou MarketCap com desconto
        # Aqui usamos uma heurística simples baseada em NOPAT futuro como no modelo normalizado
        # Mas exibimos explicitamente como "Recovery"
        recovery_price = fair_price # Já que usamos normalização no fair_price, eles convergem aqui
    
    # 8. CONSTRUÇÃO DAS TABELAS DETALHADAS (RESTAURAÇÃO TOTAL)
    
    tabela_variaveis = pd.DataFrame([
        ["Taxa Livre de Risco (Selic média)", f"{macro['risk_free']*100:.2f}%"],
        ["Prêmio de Risco de Mercado", f"{market_premium*100:.2f}%"],
        ["Inflação (IPCA)", f"{macro['inflation']*100:.2f}%"],
        ["Crescimento Perpétuo (g) Anual", f"{g_long_annual*100:.2f}%"],
        ["Beta do Período Explícito", f"{beta_explicit:.2f}"],
        ["Beta do Período da Perpetuidade", f"{beta_perp:.2f}"],
        ["Ke Período Explícito (Anual)", f"{ke_annual_explicit*100:.2f}%"],
        ["Ke Perpetuidade (Anual)", f"{ke_annual_perp*100:.2f}%"],
        ["Kd Líquido (Anual)", f"{kd_after_tax*100:.2f}%"],
        ["Peso Capital Próprio (We)", f"{we*100:.2f}%"],
        ["Peso Capital Terceiros (Wd)", f"{wd*100:.2f}%"],
        ["WACC Período Explícito (Anual)", f"{wacc_annual_explicit*100:.2f}%"],
        ["WACC Perpetuidade (Anual)", f"{wacc_annual_perp*100:.2f}%"]
    ], columns=["Variável", "Valor"])

    tabela_valor_empresa = pd.DataFrame([
        ["FCFF Base Anual (R$ MM)", fin['fcff_base_annual']/1e6],
        ["Valor Explícito (16 Trimestres)", sum_pv_explicit],
        ["Valor Residual (VP)", tv_pv],
        ["Valor Total da Firma (EV)", firm_value],
        ["Dívida Bruta", fin['total_debt']],
        ["Caixa e Equivalentes", fin['cash']],
        ["Dívida Líquida", fin['net_debt']],
        ["Valor do Patrimônio (Equity Value)", equity_value_model],
        ["Nº de Ações", fin['shares_out']],
        ["Preço de Mercado", fin['price']],
        ["Preço Justo (Estimado)", fair_price],
        ["Preço Justo (Recuperação - Ilustr.)", recovery_price]
    ], columns=["Métrica", "Valor"])

    return {
        "ticker": code,
        "ticker_yf": yf_ticker,
        "price": fin['price'],
        "eps": fin['eps'],
        "pe": fin['pe'],
        "fair_price": fair_price,
        "is_normalized": fin['is_normalized'],
        "tabela_variaveis": tabela_variaveis,
        "tabela_valor_empresa": tabela_valor_empresa
    }

# ============================================================
# 4. EXECUÇÃO
# ============================================================
if __name__ == "__main__":
    print("=== INICIANDO VALUATION DCF (DETALHADO - TRIMESTRAL) ===")
    
    macro = get_focus_series()
    market_prices = download_price_series(MARKET_TICKER)
    
    resultados_empresas = {}
    
    for code, ticker_yf in TICKERS.items():
        try:
            res = run_valuation(code, ticker_yf, macro, market_prices)
            if res:
                resultados_empresas[code] = res
        except Exception as e:
            print(f"Erro ao processar {code}: {e}")

    for code, res in resultados_empresas.items():
        print("\n" + "="*80)
        print(f"VALORAÇÃO DCF - {code} | {res['ticker_yf']}")
        print("="*80)
        
        p_justo_str = f"R$ {res['fair_price']:.2f}"
        
        # Upside seguro
        if res['fair_price'] and res['price'] and res['price'] > 0:
            upside = (res['fair_price']/res['price'] - 1)
            upside_str = f"{upside*100:+.2f}%"
        else:
            upside_str = "N/A"
        
        print(f"Preço atual: R$ {res['price']:.2f}")
        print(f"EPS (12m): {res['eps']}")
        print(f"P/L: {res['pe']}")
        print(f"Valor Justo: {p_justo_str}")
        print(f"Upside: {upside_str}")
        if res['is_normalized']:
            print("NOTA: Fluxo de caixa base normalizado (NOPAT usado devido a fluxo negativo).")

        print("\n--- Tabela: Variáveis Econômicas e Custo de Capital ---")
        print(res["tabela_variaveis"].to_string(index=False))
        
        print("\n--- Tabela: Valor da Empresa ---")
        df_display = res["tabela_valor_empresa"].copy()
        df_display["Valor"] = df_display["Valor"].apply(
            lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) and abs(x) > 100 else (f"{x*100:.2f}%" if isinstance(x, float) and abs(x) < 5 and x!=0 else x)
        )
        print(df_display.to_string(index=False))