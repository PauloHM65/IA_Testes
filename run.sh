#!/bin/bash
cd "$(dirname "$0")"

LARANJA='\033[38;5;208m'
AZUL='\033[34m'
RESET='\033[0m'

cleanup() {
    echo ""
    echo -e "${LARANJA}Encerrando Redis...${RESET}"
    docker compose down -t 3 > /dev/null 2>&1
    echo -e "${LARANJA}Até logo!${RESET}"
}
trap cleanup EXIT

echo -e "${AZUL}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${AZUL}  DRAG Qwen 2.5 + LangChain + Redis${RESET}"
echo -e "${AZUL}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""

# Redis
docker compose up -d > /dev/null 2>&1
printf "${LARANJA}⣾ Iniciando Redis...${RESET}"
until docker exec redis-rag redis-cli ping 2>/dev/null | grep -q PONG; do
    sleep 1
done
printf "\r${LARANJA}✓ Redis pronto.${RESET}              \n"

# Venv
source .venv/bin/activate

# Ingestão + Chat
echo ""
python main.py ingest
echo ""
python main.py chat
