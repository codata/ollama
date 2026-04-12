#!/bin/bash
set -e

echo "--- 🚀 STARTING PARALLEL STABILITY TEST (10 Request) ---"

# 1. Cleanup
pkill -9 ollama_indexed || true
rm -f ../ollama_parallel.log

# 2. Start server
export OLLAMA_NUM_PARALLEL=10
export OLLAMA_INDEX_METHOD=bit-signature
export OLLAMA_DEBUG=1
export OLLAMA_HOST=127.0.0.1:11435
../ollama_indexed serve > ../ollama_parallel.log 2>&1 &
SERVER_PID=$!
trap "kill -9 $SERVER_PID" EXIT

echo "--- ⏳ Waiting for server (15s)..."
sleep 15

PROMPT="create description of variable (definition, units of measurements, properties, attributes), list internal variables (units, value) and provide result in json: height"

# 3. Learning run (Sequential)
echo "--- 🧠 PHASE 1: SEQUENTIAL LEARNING ---"
curl -s -X POST http://127.0.0.1:11435/api/generate -d "{
  \"model\": \"gemma4\",
  \"prompt\": \"$PROMPT\",
  \"stream\": false
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
          \"stream\": false
        }")
        REQ_END=$(date +%s)
        REQ_ELAPSED=$((REQ_END - REQ_START))
        
        # Calculate eval rate
        E_COUNT=$(echo $RESPONSE | grep -o '"eval_count":[0-9]*' | cut -d: -f2)
        E_DUR=$(echo $RESPONSE | grep -o '"eval_duration":[0-9]*' | cut -d: -f2)
        if [ ! -z "$E_COUNT" ] && [ ! -z "$E_DUR" ] && [ "$E_DUR" -gt 0 ]; then
            EVAL_RATE=$(echo "scale=2; $E_COUNT / ($E_DUR / 1000000000)" | bc)
        fi
        echo "   <- Request #$i Finished (Eval Rate: ${EVAL_RATE:-N/A} tokens/s, Execution: ${REQ_ELAPSED} sec)"
    ) &
done

wait
END_PARALLEL=$(date +%s)
TOTAL_TIME=$((END_PARALLEL - START_PARALLEL))

echo "--- 🔍 VERIFYING STABILITY ---"
BYPASS_COUNT=$(grep -c "NITRO BYPASS" ../ollama_parallel.log)
echo "Total Nitro Bypass triggers in log: $BYPASS_COUNT"
echo "Total parallel time: ${TOTAL_TIME}s"

if [ $BYPASS_COUNT -ge 10 ]; then
    echo "✅ SUCCESS: Parallel Nitro bypass is stable!"
else
    echo "⚠️ WARNING: Some requests might have missed bypass. Count: $BYPASS_COUNT"
fi

echo "--- 🎉 TEST COMPLETED ---"
