#!/usr/bin/env bash

perf record --call-graph dwarf ./src/benchmark
perf report --tui