#!/bin/bash
set -e

echo "--- 🚀 STARTING BYPASS INTEGRATION TEST ---"

# 1. Cleanup previous runs
echo "--- 🧹 CLEANING UP ---"
pkill -9 ollama_indexed || true
rm -f ../ollama.log
rm -f bypass_test.log

# 2. Start server in background with indexing enabled
echo "--- 🚀 STARTING OLLAMA REFINED ---"
export OLLAMA_INDEX_METHOD=bit-signature
export OLLAMA_DEBUG=1
export OLLAMA_HOST=127.0.0.1:11435
../ollama_indexed serve > ../ollama.log 2>&1 &
SERVER_PID=$!

# Ensure server is killed on exit
trap "kill $SERVER_PID" EXIT

# 3. Wait for server to be ready
echo "--- ⏳ Waiting for server to start (15s)..."
sleep 15

# 4. First run: Learn session
echo "--- 🧠 PHASE 1: LEARNING ---"
START_LEARN=$(date +%s%N)
curl -s -X POST http://127.0.0.1:11435/api/generate -d '{
  "model": "gemma4",
  "prompt": "create description of variable (definition, units of measurements, properties, attributes), list internal variables (units, value) and provide result in json: height",
  "stream": false
}' > /dev/null
END_LEARN=$(date +%s%N)
ELAPSED_LEARN_MS=$((($END_LEARN - $START_LEARN) / 1000000))
ELAPSED_LEARN_S=$(echo "scale=2; $ELAPSED_LEARN_MS / 1000" | bc)
echo "Phase 1 (Learning) took: ${ELAPSED_LEARN_S}s (${ELAPSED_LEARN_MS}ms)"

echo "--- 💾 Waiting for auto-save (3s)..."
sleep 3

echo "--- ⚡ PHASE 2: MULTI-RUN BYPASSING (5 Iterations) ---"

for i in {1..5}
do
    echo "--- 🏎️ Run #$i ---"
    START_BYPASS=$(date +%s%N)
    RESPONSE=$(curl -s -X POST http://127.0.0.1:11435/api/generate -d '{
      "model": "gemma4",
      "prompt": "create description of variable (definition, units of measurements, properties, attributes), list internal variables (units, value) and provide result in json: height",
      "stream": false
    }')
    END_BYPASS=$(date +%s%N)
    
    # Extract eval rate if possible (if server returns it)
    E_COUNT=$(echo $RESPONSE | grep -o '"eval_count":[0-9]*' | cut -d: -f2)
    E_DUR=$(echo $RESPONSE | grep -o '"eval_duration":[0-9]*' | cut -d: -f2)
    if [ ! -z "$E_COUNT" ] && [ ! -z "$E_DUR" ] && [ "$E_DUR" -gt 0 ]; then
        EVAL_RATE=$(echo "scale=2; $E_COUNT / ($E_DUR / 1000000000)" | bc)
    fi
    
    ELAPSED_BYPASS_MS=$((($END_BYPASS - $START_BYPASS) / 1000000))
    ELAPSED_BYPASS_S=$(echo "scale=2; $ELAPSED_BYPASS_MS / 1000" | bc)
    
    echo "Run #$i took: ${ELAPSED_BYPASS_S}s (${ELAPSED_BYPASS_MS}ms) | Eval Rate: ${EVAL_RATE:-N/A} tokens/s"
done

echo "--- 🔍 VERIFYING LOGS ---"
BYPASS_COUNT=$(grep -c "NITRO BYPASS" ../ollama.log)
echo "Total bypass triggers found: $BYPASS_COUNT"

if [ $BYPASS_COUNT -gt 0 ]; then
    echo "✅ SUCCESS: Semantic bypass is active!"
else
    echo "❌ FAILURE: No bypass triggers found."
    exit 1
fi

echo "--- 🎉 TEST COMPLETED ---"
