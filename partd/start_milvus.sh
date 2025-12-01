#!/bin/bash
# Script to start Milvus server for Part D testing

cd /media/data/codes/reshma/lma_maj_pro/partc

# Check if Milvus is already running
if pgrep -f "milvus run" > /dev/null; then
    echo "Milvus server is already running"
    exit 0
fi

# Start Milvus standalone server
echo "Starting Milvus server..."
./bin/milvus run standalone --config configs/milvus.yaml &

# Wait for server to start
echo "Waiting for Milvus to start..."
sleep 10

# Check if server is listening
if netstat -tlnp 2>/dev/null | grep -q 19530 || ss -tlnp 2>/dev/null | grep -q 19530; then
    echo "✓ Milvus server is running on port 19530"
else
    echo "⚠ Milvus server may not be ready yet. Check logs in partc/logs/"
fi

