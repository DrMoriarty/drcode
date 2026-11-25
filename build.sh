#!/bin/bash

scripts/build-binary.sh  &&  uv run --group desktop --group build python -m desktop.packaging --target macos
