#!/bin/bash

mkdir -p ~/.streamlit/

echo "[server]
headless = true
port = \$PORT
enableXsrfProtection = false
maxUploadSize = 200

[logger]
level = \"error\"

[client]
showErrorDetails = false
" > ~/.streamlit/config.toml
