#!/bin/bash
set -e

echo "--- 🚀 STARTING DIVERSE PROMPT PERFORMANCE TEST ---"

# 1. Cleanup
pkill -9 ollama_indexed || true
lsof -ti:11435 | xargs kill -9 2>/dev/null || true
sleep 3
BASE_DIR=$(cd "$(dirname "$0")/.." && pwd)
rm -f "$BASE_DIR/ollama_diverse.log"

# 2. Start server
export OLLAMA_INDEX_METHOD=bit-signature
export OLLAMA_DEBUG=1
export OLLAMA_HOST=127.0.0.1:11435
echo "--- 🚀 Launching server from $BASE_DIR ---"
"$BASE_DIR/ollama_indexed" serve > "$BASE_DIR/ollama_diverse.log" 2>&1 &
SERVER_PID=$!
trap "kill -9 $SERVER_PID 2>/dev/null || true" EXIT

echo "--- ⏳ Waiting for server (up to 30s)..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:11435/api/version > /dev/null; then
        echo "   [✓] Server is UP"
        break
    fi
    sleep 1
done

PROMPTS=(
    "Explain the concept of quantum entanglement in simple terms."
    "Write a Python script to scrape a website and save data to CSV."
    "Summarize the history of the Roman Empire in three paragraphs."
    "What are the health benefits of a Mediterranean diet?"
    "Compare and contrast functional programming and object-oriented programming."
)

for PROMPT in "${PROMPTS[@]}"
do
    echo "---------------------------------------------------"
    echo "📝 PROMPT: $PROMPT"
    
    # Run 1: Learning
    echo "   🧠 Run 1 (Learning)..."
    START=$(date +%s%N)
    RESPONSE1=$(curl -s -X POST http://127.0.0.1:11435/api/generate -d "{
      \"model\": \"gemma4\",
      \"prompt\": \"$PROMPT\",
      \"stream\": false,
      \"options\": { \"temperature\": 0 }
    }")
    END=$(date +%s%N)
    ELAPSED=$(( ($END - $START) / 1000000 ))
    
    # Tokens/s calculation
    E_COUNT=$(echo $RESPONSE1 | grep -o '"eval_count":[0-9]*' | cut -d: -f2)
    E_DUR=$(echo $RESPONSE1 | grep -o '"eval_duration":[0-9]*' | cut -d: -f2)
    if [ ! -z "$E_COUNT" ] && [ ! -z "$E_DUR" ] && [ "$E_DUR" -gt 0 ]; then
        RATE=$(echo "scale=2; ($E_COUNT * 1000000000.0) / $E_DUR" | bc)
    fi
    
    printf "      -> Took: %.2fs | Eval Rate: %s tokens/s\n" $(echo "scale=2; $ELAPSED / 1000.0" | bc) "${RATE:-N/A}"

    echo "   💾 Waiting for auto-save (3s)..."
    sleep 3

    # Run 2: Bypassing
    echo "   ⚡ Run 2 (Bypass attempt)..."
    START=$(date +%s%N)
    RESPONSE2=$(curl -s -X POST http://127.0.0.1:11435/api/generate -d "{
      \"model\": \"gemma4\",
      \"prompt\": \"$PROMPT\",
      \"stream\": false,
      \"options\": { \"temperature\": 0 }
    }")
    END=$(date +%s%N)
    ELAPSED=$(( ($END - $START) / 1000000 ))
    
    E_COUNT=$(echo $RESPONSE2 | grep -o '"eval_count":[0-9]*' | cut -d: -f2)
    E_DUR=$(echo $RESPONSE2 | grep -o '"eval_duration":[0-9]*' | cut -d: -f2)
    if [ ! -z "$E_COUNT" ] && [ ! -z "$E_DUR" ] && [ "$E_DUR" -gt 0 ]; then
        RATE=$(echo "scale=2; ($E_COUNT * 1000000000.0) / $E_DUR" | bc)
    fi
    
    printf "      -> Took: %.2fs | Eval Rate: %s tokens/s\n" $(echo "scale=2; $ELAPSED / 1000.0" | bc) "${RATE:-N/A}"
done

echo ""
echo "--- 🔍 VERIFYING TOTAL BYPASSES ---"
BYPASS_COUNT=$(grep -ic "CODATA Fabric" "$BASE_DIR/ollama_diverse.log" || echo "0")
echo "Total CODATA Fabric hits found across all prompts: $BYPASS_COUNT"
echo "--- 🎉 TEST COMPLETED ---"
