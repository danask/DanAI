#!/bin/bash
pkill -9 -f uvicorn
lsof -ti :8000 | xargs kill -9