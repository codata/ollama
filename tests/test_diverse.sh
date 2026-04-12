#!/bin/bash
set -e

echo "--- 🚀 STARTING DIVERSE PROMPT PERFORMANCE TEST ---"

# 1. Cleanup
pkill -9 ollama_indexed || true
rm -f ../ollama_diverse.log

# 2. Start server
export OLLAMA_INDEX_METHOD=bit-signature
export OLLAMA_DEBUG=1
export OLLAMA_HOST=127.0.0.1:11435
../ollama_indexed serve > ../ollama_diverse.log 2>&1 &
SERVER_PID=$!
trap "kill -9 $SERVER_PID" EXIT

echo "--- ⏳ Waiting for server (15s)..."
sleep 15

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
      \"stream\": false
    }")
    END=$(date +%s%N)
    ELAPSED=$((($END - $START) / 1000000))
    # Tokens/s calculation
    E_COUNT=$(echo $RESPONSE1 | grep -o '"eval_count":[0-9]*' | cut -d: -f2)
    E_DUR=$(echo $RESPONSE1 | grep -o '"eval_duration":[0-9]*' | cut -d: -f2)
    RATE=$(echo "scale=2; $E_COUNT / ($E_DUR / 1000000000)" | bc)
    printf "      -> Took: %.2fs | Eval Rate: %s tokens/s\n" $(echo "scale=2; $ELAPSED / 1000" | bc) "$RATE"

    echo "   💾 Waiting for auto-save (3s)..."
    sleep 3

    # Run 2: Bypassing
    echo "   ⚡ Run 2 (Bypass attempt)..."
    START=$(date +%s%N)
    RESPONSE2=$(curl -s -X POST http://127.0.0.1:11435/api/generate -d "{
      \"model\": \"gemma4\",
      \"prompt\": \"$PROMPT\",
      \"stream\": false
    }")
    END=$(date +%s%N)
    ELAPSED=$((($END - $START) / 1000000))
    E_COUNT=$(echo $RESPONSE2 | grep -o '"eval_count":[0-9]*' | cut -d: -f2)
    E_DUR=$(echo $RESPONSE2 | grep -o '"eval_duration":[0-9]*' | cut -d: -f2)
    RATE=$(echo "scale=2; $E_COUNT / ($E_DUR / 1000000000)" | bc)
    printf "      -> Took: %.2fs | Eval Rate: %s tokens/s\n" $(echo "scale=2; $ELAPSED / 1000" | bc) "$RATE"
done

echo "--- 🔍 VERIFYING TOTAL BYPASSES ---"
BYPASS_COUNT=$(grep -c "NITRO BYPASS" ../ollama_diverse.log)
echo "Total bypass triggers found across all prompts: $BYPASS_COUNT"
echo "--- 🎉 TEST COMPLETED ---"
