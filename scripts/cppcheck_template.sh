#!/usr/bin/env bash

cppcheck --enable=all src/ test/ -I src/ -I test/ --suppress=unusedFunction --suppress=missingInclude
