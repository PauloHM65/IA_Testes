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
echo -e "${AZUL}  DRAG Multi-Serviço + LangChain + Redis${RESET}"
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

# Serviço (opcional: passar como argumento, ex: ./run.sh juridico)
SERVICE_FLAG=""
if [ -n "$1" ]; then
    SERVICE_FLAG="--service $1"
fi

# Ingestão (incremental: pula se docs não mudaram)
echo ""
python main.py $SERVICE_FLAG ingest
echo ""
python main.py $SERVICE_FLAG chat
