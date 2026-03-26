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

# Seleção de serviço
if [ -n "$1" ]; then
    SELECTED="$1"
else
    SERVICES=($(ls services/*.yaml 2>/dev/null | xargs -I{} basename {} .yaml | sort))
    if [ ${#SERVICES[@]} -le 1 ]; then
        SELECTED="${SERVICES[0]:-default}"
    else
        echo -e "${AZUL}Serviços disponíveis:${RESET}"
        for i in "${!SERVICES[@]}"; do
            echo -e "  ${LARANJA}$((i+1)))${RESET} ${SERVICES[$i]}"
        done
        echo ""
        read -p "Escolha o serviço [1-${#SERVICES[@]}]: " CHOICE
        # Valida entrada
        if [[ "$CHOICE" =~ ^[0-9]+$ ]] && [ "$CHOICE" -ge 1 ] && [ "$CHOICE" -le ${#SERVICES[@]} ]; then
            SELECTED="${SERVICES[$((CHOICE-1))]}"
        else
            echo -e "${LARANJA}Opção inválida. Usando 'default'.${RESET}"
            SELECTED="default"
        fi
    fi
fi
SERVICE_FLAG="--service $SELECTED"

# Ingestão (incremental: pula se docs não mudaram)
echo ""
python main.py $SERVICE_FLAG ingest
echo ""
python main.py $SERVICE_FLAG chat
