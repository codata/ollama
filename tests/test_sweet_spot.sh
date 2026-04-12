#!/bin/bash
set -e

run_test() {
    NUM_REQ=$1
    echo "---------------------------------------------------"
    echo "🚀 TESTING $NUM_REQ PARALLEL REQUESTS"
    echo "---------------------------------------------------"
    
    # 1. Cleanup
    pkill -9 ollama_indexed || true
    rm -f ../ollama_parallel.log

    # 2. Start server
    export OLLAMA_NUM_PARALLEL=$NUM_REQ
    export OLLAMA_INDEX_METHOD=bit-signature
    export OLLAMA_DEBUG=0
    export OLLAMA_HOST=127.0.0.1:11435
    /Users/vyacheslavtykhonov/projects/dev/ollama/ollama_indexed serve > ../ollama_parallel.log 2>&1 &
    SERVER_PID=$!
    
    echo "--- ⏳ Waiting for server (15s)..."
    sleep 15

    PROMPT="create description of variable (definition, units of measurements, properties, attributes), list internal variables (units, value) and provide result in json: height"

    # 3. Learning run
    echo "--- 🧠 PHASE 1: LEARNING ---"
    curl -s -X POST http://127.0.0.1:11435/api/generate -d "{
      \"model\": \"gemma4\",
      \"prompt\": \"$PROMPT\",
      \"stream\": false,
      \"options\": { \"temperature\": 0 }
    }" > /dev/null

    echo "--- 💾 Waiting for auto-sync (10s)..."
    sleep 10

    # 4. Parallel run
    echo "--- ⚡ PHASE 2: $NUM_REQ PARALLEL ---"
    START_B=$(date +%s)
    
    for i in $(seq 1 $NUM_REQ)
    do
        (
            REQ_START=$(date +%s)
            RESPONSE=$(curl -s -X POST http://127.0.0.1:11435/api/generate -d "{
              \"model\": \"gemma4\",
              \"prompt\": \"$PROMPT\",
              \"stream\": false,
              \"options\": { \"temperature\": 0 }
            }")
            REQ_END=$(date +%s)
            REQ_EL=$((REQ_END - REQ_START))
            E_COUNT=$(echo $RESPONSE | grep -o '"eval_count":[0-9]*' | cut -d: -f2)
            E_DUR=$(echo $RESPONSE | grep -o '"eval_duration":[0-9]*' | cut -d: -f2)
            if [ "${E_DUR:-0}" -eq 0 ]; then
                RATE="MAX (Nitro-Bypass)"
            else
                RATE=$(echo "scale=2; ($E_COUNT * 1000000000) / $E_DUR" | bc)
            fi
            echo "   <- Request #$i Finished (Eval Rate: ${RATE:-N/A} tokens/s, Execution: ${REQ_EL} sec)"
        ) &
    done

    wait
    END_B=$(date +%s)
    echo "--- ✅ Total Wall-clock: $((END_B - START_B))s ---"
    
    kill -9 $SERVER_PID || true
}

run_test 3
