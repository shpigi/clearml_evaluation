#!/usr/bin/env bash
cd /src/clearml_evaluation
git config --global --add safe.directory /src/clearml_evaluation
#jt -t grade3 -cellw 88% -T
jupyter notebook --allow-root --port=8888 --no-browser --ip=0.0.0.0
