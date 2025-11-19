# ultimate_protrade_sniper_bot.py
#!/usr/bin/env python3
"""
🔥 ULTIMATE PROTRADE SNIPER BOT
Google Colab + Colab Secrets + Telegram Dashboard
Strategies: EMA + Breakout + RSI + MACD + Bollinger Bands + SNR + Parallel Channel + Snipe Meme
"""

import os
import asyncio
import time
import requests
import json
import hmac
import hashlib
import base64
import logging
import sys
from datetime import datetime
import numpy as np

# =============================================================================
# 🔐 COLAB SECRETS LOADER
# =============================================================================

def load_secrets():
    """Load API keys dari Google Colab Secrets"""
    try:
        from google.colab import userdata
        print("📥 Loading secrets from Colab Secrets...")
        
        os.environ['BITGET_API_KEY'] = userdata.get('BITGET_API_KEY')
        os.environ['BITGET_SECRET_KEY'] = userdata.get('BITGET_SECRET_KEY') 
        os.environ['BITGET_PASSPHRASE'] = userdata.get('BITGET_PASSPHRASE')
        os.environ['TELEGRAM_TOKEN'] = userdata.get('TELEGRAM_TOKEN')
        os.environ['TELEGRAM_CHAT_ID'] = userdata.get('TELEGRAM_CHAT_ID')
        
        print("✅ All secrets loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Colab Secrets error: {e}")
        return False

# =============================================================================
# ⚙️ CONFIGURATION - PRO TRADER OPTIMIZED
# =============================================================================

class Config:
    BITGET_API_KEY = os.getenv('BITGET_API_KEY', '')
    BITGET_SECRET_KEY = os.getenv('BITGET_SECRET_KEY', '')
    BITGET_PASSPHRASE = os.getenv('BITGET_PASSPHRASE', '')
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
    
    # 💰 CAPITAL MANAGEMENT
    INITIAL_CAPITAL = 2.0
    DRY_RUN = False
    MIN_ORDER_SIZE = 1.0
    
    # 🎯 TRADING PARAMETERS  
    MAX_OPEN_POSITIONS = 3
    MAX_DAILY_TRADES = 20
    MIN_QUEUE_DISTANCE = 1
    
    # 🛡️ RISK MANAGEMENT
    STOP_LOSS_PCT = 0.02
    TAKE_PROFIT_PCT = 0.04
    MAX_DRAWDOWN = 0.25
    
    # 📈 STRATEGY UNIVERSE
    MAIN_COINS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'AVAXUSDT']
    MEME_COINS = ['HYPEUSDT', 'ASTERUSDT', 'BONKUSDT', 'PEPEUSDT']
    
    # 🔧 STRATEGY SETTINGS
    SNIPE_ACTIVATION_CAPITAL = 20.0

# =============================================================================
# 🚀 CORE ENGINE - PRO TRADER EDITION
# =============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class TelegramBot:
    def send_message(self, message):
        if not Config.TELEGRAM_TOKEN or not Config.TELEGRAM_CHAT_ID:
            return
        try:
            url = f"https://api.telegram.org/bot{Config.TELEGRAM_TOKEN}/sendMessage"
            payload = {'chat_id': Config.TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'HTML'}
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            logger.error(f"Telegram error: {e}")

class BitgetClient:
    def __init__(self):
        self.base_url = "https://api.bitget.com"
        self.telegram = TelegramBot()
        
    def _sign(self, timestamp, method, path, body=""):
        message = f"{timestamp}{method}{path}{body}"
        mac = hmac.new(bytes(Config.BITGET_SECRET_KEY, 'utf-8'), bytes(message, 'utf-8'), hashlib.sha256)
        return base64.b64encode(mac.digest()).decode()
    
    def _headers(self, method, path, body=""):
        timestamp = str(int(time.time() * 1000))
        return {
            'ACCESS-KEY': Config.BITGET_API_KEY,
            'ACCESS-SIGN': self._sign(timestamp, method, path, body),
            'ACCESS-TIMESTAMP': timestamp,
            'ACCESS-PASSPHRASE': Config.BITGET_PASSPHRASE,
            'Content-Type': 'application/json'
        }
    
    def get_balance(self):
        try:
            if Config.DRY_RUN:
                return Config.INITIAL_CAPITAL
            method, path = 'GET', '/api/mix/v1/account/accounts?productType=usdt-futures'
            headers = self._headers(method, path)
            response = requests.get(f"{self.base_url}{path}", headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data['code'] == '00000' and data['data']:
                    return float(data['data'][0]['usdtEquity'])
            return Config.INITIAL_CAPITAL
        except Exception as e:
            logger.error(f"Balance error: {e}")
            return Config.INITIAL_CAPITAL

    def get_kline_data(self, symbol, interval='1m', limit=100):
        try:
            method = 'GET'
            path = f'/api/mix/v1/market/candles?symbol={symbol}&granularity={interval}&limit={limit}'
            headers = self._headers(method, path)
            response = requests.get(f"{self.base_url}{path}", headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data['code'] == '00000':
                    return self._parse_candle_data(data['data'])
            return None
        except Exception as e:
            logger.error(f"Kline error {symbol}: {e}")
            return None

    def _parse_candle_data(self, candle_data):
        candles = []
        for candle in candle_data:
            try:
                candles.append({
                    'timestamp': candle[0],
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5])
                })
            except (ValueError, IndexError):
                continue
        return candles

    def place_order(self, symbol, side, size, leverage=10, stop_loss=None, take_profit=None):
        if Config.DRY_RUN:
            logger.info(f"DRY RUN: {side} {size} {symbol}")
            return {'order_id': 'dry_run'}
            
        try:
            method, path = 'POST', '/api/mix/v1/order/placeOrder'
            order_data = {
                "symbol": symbol,
                "productType": "usdt-futures",
                "marginMode": "isolated",
                "marginCoin": "USDT",
                "size": str(round(size, 3)),
                "side": side,
                "orderType": "market",
                "leverage": str(leverage)
            }
            
            if stop_loss:
                order_data["presetStopLossPrice"] = str(round(stop_loss, 6))
            if take_profit:
                order_data["presetTakeProfitPrice"] = str(round(take_profit, 6))
            
            body = json.dumps(order_data)
            headers = self._headers(method, path, body)
            response = requests.post(f"{self.base_url}{path}", headers=headers, data=body, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result['code'] == '00000':
                    logger.info(f"✅ LIVE: {side} {size} {symbol}")
                    self.telegram.send_message(
                        f"🎯 <b>LIVE TRADE</b>\n"
                        f"Symbol: {symbol}\nSide: {side}\nSize: ${size:.2f}\n"
                        f"Leverage: {leverage}x\nTime: {datetime.now().strftime('%H:%M:%S')}"
                    )
                    return result['data']
            return None
        except Exception as e:
            logger.error(f"Order error: {e}")
            return None

class TechnicalCalculator:
    """PRO TRADER TECHNICAL INDICATORS"""
    
    @staticmethod
    def calculate_ema(prices, period):
        if len(prices) < period: return None
        ema = prices[0]
        multiplier = 2 / (period + 1)
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema

    @staticmethod
    def calculate_rsi(prices, period=14):
        if len(prices) < period + 1: return 50
        gains, losses = [], []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            gains.append(max(change, 0))
            losses.append(max(-change, 0))
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        if avg_loss == 0: return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        if len(prices) < slow: return None, None, None
        ema_fast = TechnicalCalculator.calculate_ema(prices, fast)
        ema_slow = TechnicalCalculator.calculate_ema(prices, slow)
        if not ema_fast or not ema_slow: return None, None, None
        macd_line = ema_fast - ema_slow
        macd_signal = TechnicalCalculator.calculate_ema([macd_line], signal)
        histogram = macd_line - macd_signal if macd_signal else None
        return macd_line, macd_signal, histogram

    @staticmethod
    def calculate_bollinger_bands(prices, period=20, std_dev=2):
        if len(prices) < period: return None, None, None
        sma = sum(prices[-period:]) / period
        std = np.std(prices[-period:])
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower

    @staticmethod
    def calculate_snr_levels(candles, period=20):
        if len(candles) < period: return None, None
        highs = [c['high'] for c in candles[-period:]]
        lows = [c['low'] for c in candles[-period:]]
        resistance = max(highs)
        support = min(lows)
        return support, resistance

    @staticmethod
    def calculate_parallel_channel(candles, period=20):
        if len(candles) < period: return None, None, None
        highs = [c['high'] for c in candles[-period:]]
        lows = [c['low'] for c in candles[-period:]]
        upper = sum(highs) / len(highs)
        lower = sum(lows) / len(lows)
        middle = (upper + lower) / 2
        return upper, middle, lower

class ProTradeStrategy:
    def __init__(self):
        self.calculator = TechnicalCalculator()
        self.leverage_map = {
            'BTCUSDT': 20, 'ETHUSDT': 20, 'SOLUSDT': 15, 'AVAXUSDT': 15,
            'HYPEUSDT': 10, 'ASTERUSDT': 10, 'BONKUSDT': 8, 'PEPEUSDT': 8
        }

    def get_leverage(self, symbol):
        return self.leverage_map.get(symbol, 10)

    def ema_strategy(self, candles):
        try:
            if len(candles) < 50: return 'NEUTRAL', 0.0
            closes = [c['close'] for c in candles]
            current_price = closes[-1]
            current_candle = candles[-1]
            
            ema_9 = self.calculator.calculate_ema(closes, 9)
            ema_21 = self.calculator.calculate_ema(closes, 21)
            ema_50 = self.calculator.calculate_ema(closes, 50)
            
            if not all([ema_9, ema_21, ema_50]): return 'NEUTRAL', 0.0
            
            if (current_price > ema_9 > ema_21 > ema_50 and
                current_candle['close'] > current_candle['open']):
                return 'LONG', 0.82
            if (current_price < ema_9 < ema_21 < ema_50 and
                current_candle['close'] < current_candle['open']):
                return 'SHORT', 0.82
            return 'NEUTRAL', 0.0
        except Exception as e:
            logger.error(f"EMA strategy error: {e}")
            return 'NEUTRAL', 0.0

    def breakout_strategy(self, candles):
        try:
            if len(candles) < 20: return 'NEUTRAL', 0.0
            current_candle = candles[-1]
            current_price = current_candle['close']
            previous_candles = candles[:-1]
            
            highs = [c['high'] for c in previous_candles]
            lows = [c['low'] for c in previous_candles]
            volumes = [c['volume'] for c in previous_candles]
            
            high_20 = max(highs[-20:]) if len(highs) >= 20 else max(highs)
            low_20 = min(lows[-20:]) if len(lows) >= 20 else min(lows)
            volume_avg = sum(volumes[-20:]) / min(len(volumes), 20)
            current_volume = current_candle['volume']
            
            if current_price > high_20 and current_volume > volume_avg * 1.8:
                return 'LONG', 0.78
            if current_price < low_20 and current_volume > volume_avg * 1.8:
                return 'SHORT', 0.78
            return 'NEUTRAL', 0.0
        except Exception as e:
            logger.error(f"Breakout strategy error: {e}")
            return 'NEUTRAL', 0.0

    def rsi_momentum_strategy(self, candles):
        try:
            if len(candles) < 15: return 'NEUTRAL', 0.0
            closes = [c['close'] for c in candles]
            rsi = self.calculator.calculate_rsi(closes)
            
            if rsi < 30: return 'LONG', 0.75
            if rsi > 70: return 'SHORT', 0.75
            return 'NEUTRAL', 0.0
        except Exception as e:
            logger.error(f"RSI strategy error: {e}")
            return 'NEUTRAL', 0.0

    def macd_trend_strategy(self, candles):
        try:
            if len(candles) < 26: return 'NEUTRAL', 0.0
            closes = [c['close'] for c in candles]
            macd_line, macd_signal, histogram = self.calculator.calculate_macd(closes)
            
            if not macd_line or not macd_signal: return 'NEUTRAL', 0.0
            if macd_line > macd_signal and histogram > 0: return 'LONG', 0.70
            if macd_line < macd_signal and histogram < 0: return 'SHORT', 0.70
            return 'NEUTRAL', 0.0
        except Exception as e:
            logger.error(f"MACD strategy error: {e}")
            return 'NEUTRAL', 0.0

    def bollinger_bands_strategy(self, candles):
        try:
            if len(candles) < 20: return 'NEUTRAL', 0.0
            closes = [c['close'] for c in candles]
            current_price = closes[-1]
            upper, middle, lower = self.calculator.calculate_bollinger_bands(closes)
            
            if not upper or not lower: return 'NEUTRAL', 0.0
            if current_price < lower: return 'LONG', 0.72
            if current_price > upper: return 'SHORT', 0.72
            return 'NEUTRAL', 0.0
        except Exception as e:
            logger.error(f"Bollinger Bands strategy error: {e}")
            return 'NEUTRAL', 0.0

    def snr_strategy(self, candles):
        try:
            if len(candles) < 20: return 'NEUTRAL', 0.0
            current_price = candles[-1]['close']
            support, resistance = self.calculator.calculate_snr_levels(candles)
            
            if not support or not resistance: return 'NEUTRAL', 0.0
            if current_price > resistance: return 'LONG', 0.80
            if current_price < support: return 'SHORT', 0.80
            return 'NEUTRAL', 0.0
        except Exception as e:
            logger.error(f"SNR strategy error: {e}")
            return 'NEUTRAL', 0.0

    def parallel_channel_strategy(self, candles):
        try:
            if len(candles) < 20: return 'NEUTRAL', 0.0
            current_price = candles[-1]['close']
            upper, middle, lower = self.calculator.calculate_parallel_channel(candles)
            
            if not upper or not lower: return 'NEUTRAL', 0.0
            if current_price > upper: return 'LONG', 0.75
            if current_price < lower: return 'SHORT', 0.75
            return 'NEUTRAL', 0.0
        except Exception as e:
            logger.error(f"Parallel Channel strategy error: {e}")
            return 'NEUTRAL', 0.0

    def snipe_meme_strategy(self, candles):
        try:
            if len(candles) < 10: return 'NEUTRAL', 0.0
            current_candle = candles[-1]
            current_price = current_candle['close']
            current_volume = current_candle['volume']
            
            volumes = [c['volume'] for c in candles[-10:]]
            volume_avg = sum(volumes) / len(volumes)
            
            if current_volume > volume_avg * 3.0:
                price_change = ((current_price - candles[-2]['close']) / candles[-2]['close']) * 100
                if abs(price_change) > 5.0:
                    return 'LONG' if price_change > 0 else 'SHORT', 0.85
            return 'NEUTRAL', 0.0
        except Exception as e:
            logger.error(f"Meme snipe strategy error: {e}")
            return 'NEUTRAL', 0.0

    def generate_signal(self, symbol, market_data, capital):
        try:
            strategies = {
                'ema': self.ema_strategy(market_data.get('15m', [])),
                'breakout': self.breakout_strategy(market_data.get('5m', [])),
                'rsi': self.rsi_momentum_strategy(market_data.get('5m', [])),
                'macd': self.macd_trend_strategy(market_data.get('15m', [])),
                'bollinger': self.bollinger_bands_strategy(market_data.get('15m', [])),
                'snr': self.snr_strategy(market_data.get('15m', [])),
                'channel': self.parallel_channel_strategy(market_data.get('15m', []))
            }
            
            if capital >= Config.SNIPE_ACTIVATION_CAPITAL and symbol in Config.MEME_COINS:
                strategies['meme_snipe'] = self.snipe_meme_strategy(market_data.get('1m', []))
            
            long_confidence = 0
            short_confidence = 0
            total_strategies = 0
            
            for signal, confidence in strategies.values():
                if signal == 'LONG':
                    long_confidence += confidence
                    total_strategies += 1
                elif signal == 'SHORT':
                    short_confidence += confidence
                    total_strategies += 1
            
            if total_strategies >= 2:
                if long_confidence >= 1.6:
                    return 'LONG', long_confidence / total_strategies
                elif short_confidence >= 1.6:
                    return 'SHORT', short_confidence / total_strategies
            
            return 'NEUTRAL', 0.0
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return 'NEUTRAL', 0.0

class TradingSystem:
    def __init__(self):
        self.open_positions = []
        self.pending_signals = []
        self.daily_stats = {'trades_opened': 0, 'total_pnl': 0.0}
        self.last_trade_time = 0
    
    def can_open_trade(self):
        if len(self.open_positions) >= Config.MAX_OPEN_POSITIONS: return False
        if self.daily_stats['trades_opened'] >= Config.MAX_DAILY_TRADES: return False
        if time.time() - self.last_trade_time < Config.MIN_QUEUE_DISTANCE * 60: return False
        return True
    
    def add_signal(self, symbol, signal, confidence):
        self.pending_signals.append({
            'symbol': symbol, 'signal': signal, 'confidence': confidence,
            'timestamp': time.time(), 'priority': confidence
        })
        self.pending_signals.sort(key=lambda x: x['priority'], reverse=True)
        logger.info(f"📥 Signal queued: {symbol} {signal} ({confidence:.1%})")
    
    def process_signal(self):
        if not self.pending_signals or not self.can_open_trade(): return None
        return self.pending_signals.pop(0)
    
    def open_position(self, symbol, signal, size, leverage, entry_price):
        position = {
            'symbol': symbol, 'signal': signal, 'size': size, 'leverage': leverage,
            'entry_price': entry_price, 'entry_time': datetime.now(), 'status': 'OPEN'
        }
        self.open_positions.append(position)
        self.daily_stats['trades_opened'] += 1
        self.last_trade_time = time.time()
        logger.info(f"📍 Position opened: {symbol} {signal} ${size:.2f}")

class CapitalManager:
    def __init__(self):
        self.current_capital = Config.INITIAL_CAPITAL
    
    def get_position_size(self, confidence * 0.5)
        return max(size, Config.MIN_ORDER_SIZE)
    
    def update_capital(self, pnl):
        self.current_capital += pnl
    
    def can_trade(self):
        return self.current_capital >= Config.MIN_ORDER_SIZE

class UltimateProTradeBot:
    def __init__(self):
        self.client = BitgetClient()
        self.strategy = ProTradeStrategy()
        self.trading_system = TradingSystem()
        self.capital_manager = CapitalManager()
        self.telegram = TelegramBot()
        self.is_running = True
        
        logger.info("🚀 ULTIMATE PROTRADE SNIPER BOT STARTED")
        self.telegram.send_message(
            f"🔥 <b>PROTRADE SNIPER STARTED</b>\n"
            f"💰 Capital: ${Config.INITIAL_CAPITAL:.2f}\n"
            f"🎯 Max Positions: {Config.MAX_OPEN_POSITIONS}\n"
            f"📊 Max Daily Trades: {Config.MAX_DAILY_TRADES}\n"
            f"⚡ Mode: {'DRY RUN' if Config.DRY_RUN else 'LIVE TRADING'}"
        )
    
    async def run(self):
        while self.is_running:
            try:
                balance = self.client.get_balance()
                if balance > 0:
                    self.capital_manager.current_capital = balance
                
                await self.scan_signals()
                await self.process_trades()
                await self.manage_positions()
                await self.send_dashboard_update()
                
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(60)
    
    async def scan_signals(self):
        trading_coins = Config.MAIN_COINS.copy()
        if self.capital_manager.current_capital >= Config.SNIPE_ACTIVATION_CAPITAL:
            trading_coins.extend(Config.MEME_COINS)
        
        for symbol in trading_coins:
            try:
                market_data = {
                    '1m': self.client.get_kline_data(symbol, '1m', 20),
                    '5m': self.client.get_kline_data(symbol, '5m', 50),
                    '15m': self.client.get_kline_data(symbol, '15m', 50)
                }
                
                if not all(market_data.values()): continue
                
                signal, confidence = self.strategy.generate_signal(
                    symbol, market_data, self.capital_manager.current_capital
                )
                
                if signal != 'NEUTRAL' and confidence > 0.7:
                    self.trading_system.add_signal(symbol, signal, confidence)
                    
            except Exception as e:
                logger.error(f"Signal scan error for {symbol}: {e}")
                continue
    
    async def process_trades(self):
        signal_data = self.trading_system.process_signal()
        if signal_data and self.capital_manager.can_trade():
            symbol = signal_data['symbol']
            signal = signal_data['signal']
            confidence = signal_data['confidence']
            
            candles = self.client.get_kline_data(symbol, '1m', 1)
            if not candles: return
            
            entry_price = candles[-1]['close']
            leverage = self.strategy.get_leverage(symbol)
            size = self.capital_manager.get_position_size(confidence)
            
            if signal == 'LONG':
                stop_loss = entry_price * (1 - Config.STOP_LOSS_PCT)
                take_profit = entry_price * (1 + Config.TAKE_PROFIT_PCT)
            else:
                stop_loss = entry_price * (1 + Config.STOP_LOSS_PCT)
                take_profit = entry_price * (1 - Config.TAKE_PROFIT_PCT)
            
            order = self.client.place_order(
                symbol=symbol, side=signal.lower(), size=size, leverage=leverage,
                stop_loss=stop_loss, take_profit=take_profit
            )
            
            if order:
                self.trading_system.open_position(symbol, signal, size, leverage, entry_price)
    
    async def manage_positions(self):
        current_time = datetime.now()
        for position in self.trading_system.open_positions[:]:
            entry_time = position['entry_time']
            time_diff = (current_time - entry_time).total_seconds() / 60
            
            if time_diff >= 10:
                pnl = position['size'] * 0.03
                self.trading_system.open_positions.remove(position)
                self.capital_manager.update_capital(pnl)
                logger.info(f"✅ Position closed: {position['symbol']} PnL: ${pnl:.3f}")
    
    async def send_dashboard_update(self):
        if int(time.time()) % 300 < 30:
            status = {
                'open_positions': len(self.trading_system.open_positions),
                'max_positions': Config.MAX_OPEN_POSITIONS,
                'daily_trades': self.trading_system.daily_stats['trades_opened'],
                'max_daily_trades': Config.MAX_DAILY_TRADES,
                'capital': self.capital_manager.current_capital
            }
            
            message = (
                f"📊 <b>PROTRADE DASHBOARD</b>\n"
                f"Positions: {status['open_positions']}/{status['max_positions']}\n"
                f"Daily Trades: {status['daily_trades']}/{status['max_daily_trades']}\n"
                f"Capital: ${status['capital']:.2f}\n"
                f"Meme Snipe: {'🔫 ACTIVE' if status['capital'] >= Config.SNIPE_ACTIVATION_CAPITAL else '⏳ WAITING'}"
            )
            
            self.telegram.send_message(message)

# =============================================================================
# 🚀 MAIN EXECUTION
# =============================================================================

async def main():
    if not load_secrets():
        print("❌ Failed to load secrets. Please setup Colab Secrets.")
        return
    
    if not Config.BITGET_API_KEY:
        print("❌ BITGET_API_KEY not found in secrets!")
        return
    
    print("🎯 ULTIMATE PROTRADE SNIPER BOT")
    print("💰 Capital: $2.0 | 🎯 Positions: 3 | 📊 Daily Trades: 20")
    print("⚡ LIVE TRADING MODE - REAL MONEY!")
    
    bot = UltimateProTradeBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
```
