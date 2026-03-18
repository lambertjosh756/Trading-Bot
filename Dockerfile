FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy bot files
COPY crypto_trader.py .
COPY crypto_scalper.py .
COPY crypto_compare.py .
COPY logger.py* ./

# Logs and dashboard written to /app at runtime
ENV PYTHONUTF8=1

# Run trend-follower (port 8080) and scalper (port 8081) side-by-side.
# Fly.io exposes 8080; scalper_dashboard.html is also served there.
# If either process exits the container restarts automatically.
CMD ["sh", "-c", "python -u crypto_trader.py & python -u crypto_scalper.py --capital 10000 & python -u crypto_compare.py --watch & wait"]
