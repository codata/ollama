#!/bin/bash
set -e

echo "--- 🚀 STARTING PARALLEL STABILITY TEST (10 Request) ---"

# 1. Cleanup - Crucial to clear VRAM before starting a new compute-intensive test
pkill -9 ollama_indexed || true
sleep 3
rm -f ../ollama_parallel.log

# 2. Start server
BASE_DIR=$(cd "$(dirname "$0")/.." && pwd)
export OLLAMA_NUM_PARALLEL=10
export OLLAMA_INDEX_METHOD=bit-signature
export OLLAMA_DEBUG=1
export OLLAMA_HOST=127.0.0.1:11435
echo "--- 🚀 Launching server from $BASE_DIR ---"
"$BASE_DIR/ollama_indexed" serve > "$BASE_DIR/ollama_parallel.log" 2>&1 &
SERVER_PID=$!
trap "kill -9 $SERVER_PID 2>/dev/null || true" EXIT

echo "--- ⏳ Waiting for server (15s)..."
for i in {1..15}; do
    if curl -s http://127.0.0.1:11435/api/version > /dev/null; then
        echo "   [✓] Server is UP"
        break
    fi
    sleep 1
done

PROMPT="create description of variable (definition, units of measurements, properties, attributes), list internal variables (units, value) and provide result in json: height"

# 3. Learning run (Sequential)
echo "--- 🧠 PHASE 1: SEQUENTIAL LEARNING ---"
curl -s -X POST http://127.0.0.1:11435/api/generate -d "{
  \"model\": \"gemma4\",
  \"prompt\": \"$PROMPT\",
  \"stream\": false,
  \"options\": { \"temperature\": 0 }
}" > /dev/null

echo "--- 💾 Waiting for auto-save (3s)..."
sleep 3

# 4. Parallel run
echo "--- ⚡ PHASE 2: 10 PARALLEL REQUESTS ---"
START_PARALLEL=$(date +%s)

for i in {1..10}
do
    (
        echo "   -> Starting Request #$i"
        REQ_START=$(date +%s)
        RESPONSE=$(curl -s -X POST http://127.0.0.1:11435/api/generate -d "{
          \"model\": \"gemma4\",
          \"prompt\": \"$PROMPT\",
          \"stream\": false,
          \"options\": { \"temperature\": 0 }
        }")
        REQ_END=$(date +%s)
        REQ_ELAPSED=$((REQ_END - REQ_START))
        
        # Calculate eval rates
        E_COUNT=$(echo $RESPONSE | grep -o '"eval_count":[0-9]*' | cut -d: -f2)
        E_DUR=$(echo $RESPONSE | grep -o '"eval_duration":[0-9]*' | cut -d: -f2)
        PE_COUNT=$(echo $RESPONSE | grep -o '"prompt_eval_count":[0-9]*' | cut -d: -f2)
        PE_DUR=$(echo $RESPONSE | grep -o '"prompt_eval_duration":[0-9]*' | cut -d: -f2)

        if [ ! -z "$E_COUNT" ] && [ ! -z "$E_DUR" ] && [ "$E_DUR" -gt 0 ]; then
            EVAL_RATE=$(echo "scale=2; ($E_COUNT * 1000000000.0) / $E_DUR" | bc)
        fi
        if [ ! -z "$PE_COUNT" ] && [ ! -z "$PE_DUR" ] && [ "$PE_DUR" -gt 0 ]; then
            PE_RATE=$(echo "scale=2; ($PE_COUNT * 1000000000.0) / $PE_DUR" | bc)
        fi

        echo "   <- Request #$i Finished (Prompt: ${PE_RATE:-N/A} t/s | Eval: ${EVAL_RATE:-N/A} t/s | Time: ${REQ_ELAPSED}s)"
    ) &
done

wait
END_PARALLEL=$(date +%s)
TOTAL_TIME=$((END_PARALLEL - START_PARALLEL))

echo ""
echo "--- 🔍 VERIFYING TOTAL BYPASSES ---"
BYPASS_COUNT=$(grep -ic "CODATA Fabric" "$BASE_DIR/ollama_parallel.log" || echo "0")
echo "Total CODATA Fabric Bypass hits found: $BYPASS_COUNT"
echo "Total parallel time: ${TOTAL_TIME}s"

if [ $BYPASS_COUNT -ge 10 ]; then
    echo "✅ SUCCESS: Parallel CODATA Fabric Bypass is stable!"
else
    echo "⚠️ WARNING: Some requests might have missed bypass. Count: $BYPASS_COUNT"
fi

echo "--- 🎉 TEST COMPLETED ---"
