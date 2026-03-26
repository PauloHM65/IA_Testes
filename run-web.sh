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
echo -e "${AZUL}  DRAG Web — LangChain + Redis${RESET}"
echo -e "${AZUL}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""

# Redis
docker compose up redis -d > /dev/null 2>&1
printf "${LARANJA}⣾ Iniciando Redis...${RESET}"
until docker exec redis-rag redis-cli ping 2>/dev/null | grep -q PONG; do
    sleep 1
done
printf "\r${LARANJA}✓ Redis pronto.${RESET}              \n"

# Venv
source .venv/bin/activate

# Ingestão de todos os serviços
for yaml_file in services/*.yaml; do
    SERVICE=$(basename "$yaml_file" .yaml)
    echo -e "${LARANJA}Ingestão: ${SERVICE}${RESET}"
    python main.py --service "$SERVICE" ingest
done

# API + Frontend
echo ""
echo -e "${AZUL}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${LARANJA}  Abra no browser: http://localhost:8080${RESET}"
echo -e "${AZUL}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
uvicorn api:app --host 0.0.0.0 --port 8080
