import asyncio
import random
import time
from typing import AsyncIterator, List, Dict, Any, Tuple, Optional
from collections import deque
from dataclasses import dataclass
import logging
from enum import Enum, auto

# --- Constants and Configuration ---
class PredictionResult(Enum):
    UP = auto()
    DOWN = auto()
    NEUTRAL = auto()
    WAITING_FOR_DATA = auto()
    ERROR = auto()

@dataclass
class StockConfig:
    symbols: List[str]
    initial_price: float = 100.0
    volatility: float = 2.0
    update_interval: float = 0.5
    min_price: float = 1.0
    history_size: int = 10

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('stock_predictor.log')
    ]
)
logger = logging.getLogger(_name_)

# --- Exception Handling ---
class StockPredictionError(Exception):
    """Base exception for stock prediction errors"""
    pass

class InsufficientDataError(StockPredictionError):
    """Raised when there's not enough data for prediction"""
    pass

# --- Database Connection with Connection Pooling ---
class AsyncDatabaseManager:
    """Improved database connection manager with connection pooling"""
    def _init_(self, db_url: str, pool_size: int = 5):
        self.db_url = db_url
        self.pool_size = pool_size
        self._connection_pool = None

    async def _aenter_(self):
        # Simulate connection pool creation
        await asyncio.sleep(0.1)
        self._connection_pool = [f"Connection-{i}" for i in range(self.pool_size)]
        logger.info(f"Created connection pool to {self.db_url} with {self.pool_size} connections")
        return self

    async def _aexit_(self, exc_type, exc_val, exc_tb):
        # Clean up connections
        await asyncio.sleep(0.1)
        self._connection_pool = None
        if exc_type:
            logger.error(f"Database error: {exc_val}")
        logger.info("Closed database connection pool")

    async def execute_query(self, query: str) -> Any:
        """Execute a database query using connection from pool"""
        if not self._connection_pool:
            raise StockPredictionError("Database connection pool not initialized")
        
        # Get a connection from the pool
        conn = self._connection_pool.pop()
        try:
            logger.debug(f"Executing query: {query}")
            await asyncio.sleep(0.05)  # Simulate query execution
            return f"Result for {query}"
        finally:
            # Return connection to pool
            self._connection_pool.append(conn)

# --- Enhanced Stock Feed with Market Events ---
class StockFeed:
    """Enhanced stock feed simulator with market events"""
    def _init_(self, config: StockConfig):
        self.config = config
        self.prices: Dict[str, float] = {s: config.initial_price for s in config.symbols}
        self._market_open = True
        self._last_update = time.time()

    async def market_hours_check(self):
        """Simulate market open/close cycles"""
        while True:
            await asyncio.sleep(3600)  # Check every hour
            self._market_open = not self._market_open
            status = "open" if self._market_open else "closed"
            logger.info(f"Market status changed to {status}")

    async def generate_feed(self) -> AsyncIterator[Dict[str, Any]]:
        """Generate stock data with market events"""
        market_task = asyncio.create_task(self.market_hours_check())
        
        try:
            while True:
                await asyncio.sleep(self.config.update_interval)
                if not self._market_open:
                    yield {"type": "market_event", "status": "closed"}
                    continue

                timestamp = time.time()
                for symbol in self.config.symbols:
                    # More realistic price movement simulation
                    change = random.gauss(0, self.config.volatility/2)
                    self.prices[symbol] += change
                    self.prices[symbol] = max(self.config.min_price, self.prices[symbol])
                    
                    yield {
                        "type": "stock_data",
                        "symbol": symbol,
                        "price": round(self.prices[symbol], 2),
                        "timestamp": timestamp,
                        "volume": random.randint(1000, 10000)  # Simulate volume
                    }
                self._last_update = timestamp
        finally:
            market_task.cancel()

# --- Machine Learning Predictor with Model Loading ---
class StockPredictor:
    """Enhanced stock predictor with model loading simulation"""
    def _init_(self, config: StockConfig):
        self.config = config
        self.history: Dict[str, deque] = {s: deque(maxlen=config.history_size) for s in config.symbols}
        self._model = self._load_model()
        logger.info("Stock predictor initialized")

    def _load_model(self) -> str:
        """Simulate model loading"""
        # In real implementation, this would load TensorFlow/PyTorch model
        logger.info("Loading prediction model...")
        time.sleep(0.5)  # Simulate model loading time
        return "pretrained_lstm_model"

    def update_data(self, data: Dict[str, Any]) -> None:
        """Update historical data"""
        if data["type"] != "stock_data":
            return
            
        symbol = data["symbol"]
        self.history[symbol].append({
            "price": data["price"],
            "volume": data["volume"],
            "timestamp": data["timestamp"]
        })

    async def predict(self, symbol: str) -> Tuple[str, PredictionResult]:
        """Make prediction for a given symbol"""
        try:
            if len(self.history[symbol]) < self.config.history_size:
                raise InsufficientDataError(f"Not enough data for {symbol}")
            
            # Simulate model prediction
            await asyncio.sleep(0.1)  # Model inference time
            
            # More sophisticated dummy prediction
            history = list(self.history[symbol])
            prices = [h["price"] for h in history]
            avg_price = sum(prices) / len(prices)
            last_price = prices[-1]
            
            if last_price > avg_price * 1.02:
                return symbol, PredictionResult.UP
            elif last_price < avg_price * 0.98:
                return symbol, PredictionResult.DOWN
            else:
                return symbol, PredictionResult.NEUTRAL
                
        except Exception as e:
            logger.error(f"Prediction failed for {symbol}: {str(e)}")
            return symbol, PredictionResult.ERROR

# --- Prediction Analytics ---
class PredictionTracker:
    """Tracks prediction accuracy and performance"""
    def _init_(self):
        self.predictions: Dict[str, List[Tuple[float, PredictionResult]]] = {}
        self.performance_stats = {
            "total_predictions": 0,
            "correct_predictions": 0
        }

    def record_prediction(self, symbol: str, price: float, prediction: PredictionResult):
        if symbol not in self.predictions:
            self.predictions[symbol] = []
        self.predictions[symbol].append((price, prediction))
        self.performance_stats["total_predictions"] += 1

    def calculate_accuracy(self) -> Dict[str, float]:
        """Calculate prediction accuracy per symbol"""
        accuracy = {}
        for symbol, preds in self.predictions.items():
            if len(preds) < 2:
                continue
                
            correct = 0
            for i in range(len(preds)-1):
                current_price = preds[i][0]
                next_price = preds[i+1][0]
                prediction = preds[i][1]
                
                if (prediction == PredictionResult.UP and next_price > current_price) or \
                   (prediction == PredictionResult.DOWN and next_price < current_price):
                    correct += 1
                    self.performance_stats["correct_predictions"] += 1
            
            accuracy[symbol] = correct / (len(preds)-1) if len(preds) > 1 else 0.0
        
        return accuracy

# --- Main Application ---
class StockPredictionApp:
    """Main application class"""
    def _init_(self, config: StockConfig):
        self.config = config
        self.feed = StockFeed(config)
        self.predictor = StockPredictor(config)
        self.tracker = PredictionTracker()
        self.db_config = "postgresql://user:pass@localhost/stocks"
        self.running = False

    async def send_to_dashboard(self, prediction_data: Dict[str, Any]):
        """Simulate sending data to dashboard"""
        logger.debug(f"Sending to dashboard: {prediction_data}")
        await asyncio.sleep(0.02)  # Network delay simulation

    async def run(self):
        """Run the prediction service"""
        self.running = True
        logger.info("Starting Stock Prediction Service")
        
        async with AsyncDatabaseManager(self.db_config) as db:
            feed_task = asyncio.create_task(self.feed.generate_feed())
            
            try:
                async for data in feed_task:
                    if not self.running:
                        break
                        
                    if data["type"] == "market_event":
                        logger.info(f"Market is now {data['status']}")
                        continue
                        
                    # Process stock data
                    symbol = data["symbol"]
                    logger.info(f"Processing {symbol} @ {data['price']:.2f}")
                    
                    # Update predictor
                    self.predictor.update_data(data)
                    
                    # Make prediction
                    symbol, prediction = await self.predictor.predict(symbol)
                    if prediction not in [PredictionResult.WAITING_FOR_DATA, PredictionResult.ERROR]:
                        logger.info(f"Prediction for {symbol}: {prediction.name}")
                        
                        # Track prediction
                        self.tracker.record_prediction(symbol, data["price"], prediction)
                        
                        # Store in database
                        await db.execute_query(
                            f"INSERT INTO predictions VALUES ('{symbol}', {data['price']}, '{prediction.name}')"
                        )
                        
                        # Send to dashboard
                        await self.send_to_dashboard({
                            "symbol": symbol,
                            "price": data["price"],
                            "prediction": prediction.name,
                            "timestamp": time.time()
                        })
                    
            except asyncio.CancelledError:
                logger.info("Service shutdown requested")
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
            finally:
                self.running = False
                feed_task.cancel()
                
                # Calculate and log accuracy before exiting
                accuracy = self.tracker.calculate_accuracy()
                for symbol, acc in accuracy.items():
                    logger.info(f"Prediction accuracy for {symbol}: {acc:.2%}")
                
                logger.info("Service shutdown complete")

    async def shutdown(self):
        """Graceful shutdown"""
        self.running = False
        logger.info("Initiating graceful shutdown...")

if _name_ == "_main_":
    config = StockConfig(
        symbols=["AAPL", "GOOGL", "MSFT", "AMZN"],
        initial_price=150.0,
        volatility=3.0,
        history_size=15
    )
    
    app = StockPredictionApp(config)
    
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        asyncio.run(app.shutdown())
    except Exception as e:
        logger.error(f"Application error: {str(e)}")