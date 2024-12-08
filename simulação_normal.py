import pandas as pd
import numpy as np
import ccxt
import yfinance as yf
import matplotlib.pyplot as plt


def fetch_bitcoin_data(exchange: ccxt.Exchange, symbol: str, limit: int = 365) -> pd.DataFrame:
    """Busca dados históricos do Bitcoin."""
    ohlcv = exchange.fetch_ohlcv(symbol, '1d', limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize(None)
    df.set_index('timestamp', inplace=True)
    return df


def calculate_expected_return(beta: float, mrp: float, rf_rate: float) -> float:
    """Calcula a taxa de retorno esperada com base no modelo CAPM."""
    return rf_rate + beta * mrp


def calculate_annualized_volatility(df: pd.DataFrame, volatility_days: int = 90) -> float:
    """Calcula a volatilidade anualizada com base em uma janela móvel."""
    df['pct_change'] = df['close'].pct_change() * 100
    df['stdev'] = df['pct_change'].rolling(volatility_days).std()
    df['vol'] = df['stdev'] * (365 ** 0.5)
    return df['vol'].mean() / 100


def simulate_prices(S0: float, r: float, sigma: float, M: int, I: int) -> np.ndarray:
    """Simula os preços futuros do Bitcoin usando Monte Carlo."""
    dt = 1 / M
    S = S0 * np.exp(np.cumsum(
        (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.standard_normal((M + 1, I)), axis=0))
    S[0] = S0
    return S


if __name__ == "__main__":
    # Configurações iniciais
    exchange = ccxt.binance()
    symbol = 'BTC/USDT'
    target_date = '2024-12-05'

    # Busca os dados do Bitcoin
    df = fetch_bitcoin_data(exchange, symbol)

    # Obtém o preço inicial (fechamento do Bitcoin na data alvo)
    try:
        target_date_parsed = pd.to_datetime(target_date)
        df.index = pd.to_datetime(df.index)  # Garante que o índice é datetime
        S0 = df.loc[target_date_parsed, 'close']
    except Exception as e:
        raise ValueError(f"Erro ao buscar o preço inicial para {target_date}: {e}")

    print(f"S0 (preço inicial em {target_date}): {S0}")

    # Cálculo de volatilidade anualizada
    sigma = calculate_annualized_volatility(df)

    # Obtém dados do mercado (S&P 500 e taxa livre de risco)
    sp500 = yf.Ticker("^GSPC")
    sp500_returns = sp500.history(period="5y")['Close'].pct_change().dropna()

    # Tornar ambos os índices timezone-naive
    sp500_returns.index = sp500_returns.index.tz_localize(None)

    # Calcula os retornos diários do Bitcoin
    btc_returns = df['close'].pct_change().dropna()

    # Alinha os retornos do Bitcoin e do S&P 500
    aligned_data = pd.DataFrame({
        'btc_returns': btc_returns,
        'sp500_returns': sp500_returns
    }).dropna()

    if aligned_data.empty:
        raise ValueError("Os dados alinhados entre BTC e S&P 500 estão vazios. Verifique os intervalos de datas.")

    # Calcula o Beta
    beta = np.cov(aligned_data['btc_returns'], aligned_data['sp500_returns'])[0, 1] / np.var(aligned_data['sp500_returns'])

    # Taxa livre de risco
    irx = yf.Ticker("^IRX")
    rf_rate = irx.history(period="1d")['Close'].iloc[-1] / 100

    # Prêmio de risco do mercado
    expected_market_return = sp500_returns.mean() * 252
    mrp = expected_market_return - rf_rate

    # Calcula o retorno esperado
    r = calculate_expected_return(beta, mrp, rf_rate)

    # Simula os preços futuros do Bitcoin
    M, I = 365, 50000  # Parâmetros: dias e trajetórias
    S = simulate_prices(S0, r, sigma, M, I)

    # Preço real do BTC (últimos M dias)
    real_prices = df['close'][-M:]

    # Visualiza os preços simulados
    plt.figure(figsize=(18, 8))
    plt.title('Simulação Monte Carlo para Bitcoin (Último Ano)')
    plt.plot(real_prices.values, 'k+', label="Preço Real do BTC")  # Adiciona o preço real como pontos
    plt.plot(S[:, :20], alpha=0.7)
    plt.grid(True)
    plt.xlabel('Dias')
    plt.ylabel('BTC (USD)')
    plt.legend().remove()  # Remove a legenda
    plt.show()

    # Calcula estatísticas das simulações
    closing_prices = S[-1]
    top_ten = np.percentile(closing_prices, 90)
    bottom_ten = np.percentile(closing_prices, 10)

    # Visualiza a distribuição dos preços simulados
    plt.figure(figsize=(18, 8))
    plt.hist(closing_prices, bins=100, alpha=0.7, label='Preços Simulados')
    plt.axvline(top_ten, color='r', linestyle='dashed', linewidth=2)  # Percentil 95%
    plt.axvline(bottom_ten, color='purple', linestyle='dashed', linewidth=2)  # Percentil 5% em roxo
    plt.xlabel('Preço do BTC')
    plt.ylabel('Frequência')
    plt.legend().remove()  # Remove a legenda
    plt.show()

    # Exibe as métricas finais
    print("Resultados da Simulação:")
    print("-------------------------")
    print(f"Preço Médio Esperado do BTC: \t {round(np.mean(closing_prices), 2)}")
    print(f"Percentil 5%: \t\t\t {np.percentile(closing_prices, 5)}")
    print(f"Percentil 95%: \t\t\t {np.percentile(closing_prices, 95)}")
    print(f"Nº de Simulações: \t\t {I}")
    below_price_today = (closing_prices < S0).sum()
    print(f"Nº de Simulações Abaixo do Preço Atual: \t {below_price_today}")
    print(f"Percentual de Simulações Abaixo: \t {below_price_today * 100 / I:.2f}%")

    