"""Gera data/tarefa_ciencia.txt — corpus-tarefa de estilo técnico-científico pt-BR.

Frases escritas para este lab (originais, sem restrição de direitos); parágrafos
montados com ordem determinística (semente fixa) para reproduzibilidade.
Uso: python scripts/prepara_tarefa_ciencia.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "data" / "tarefa_ciencia.txt"

FRASES = [
    "O sensor mediu a temperatura ambiente com precisão de zero vírgula um grau Celsius.",
    "A amostra foi analisada por espectrometria de massa em modo de ionização positiva.",
    "O protocolo experimental exige três repetições independentes por condição testada.",
    "Os dados brutos foram registrados no laboratório de computação científica às dezenove horas.",
    "A calibração do equipamento segue a norma internacional vigente desde o ano passado.",
    "O modelo estatístico assumiu distribuição normal dos resíduos e variância constante.",
    "A experiência produziu um aumento significativo de rendimento em relação ao grupo de controle.",
    "Os reagentes foram armazenados a menos vinte graus Celsius para evitar degradação.",
    "O circuito integrado dissipa calor proporcional ao quadrado da corrente elétrica.",
    "A medição do campo magnético foi feita com sonda Hall calibrada em laboratório credenciado.",
    "O algoritmo convergiu em mil e duzentas iterações com erro relativo abaixo do limiar.",
    "A rede de sensores transmitiu os pacotes em intervalo de um segundo por nós ativos.",
    "Os resultados sugerem correlação positiva entre a concentração e a taxa de reação.",
    "O espectro de absorção apresentou pico em quatrocentos e oitenta nanômetros.",
    "A incerteza combinada foi estimada conforme o guia para a expressão de incerteza de medição.",
    "O ensaio mecânico revelou limite de escoamento acima do valor especificado na norma.",
    "A simulação numérica utilizou diferenças finitas com malha refinada na região de interesse.",
    "Os parâmetros hiperbólicos foram ajustados por validação cruzada em dez partes.",
    "A microscopia eletrônica de varredura mostrou porosidade homogênea na superfície fraturada.",
    "O conversor analógico digital amostrou o sinal a cem quilohertz com resolução de dezesseis bits.",
    "A curva de calibração exibiu linearidade entre zero e cinco miligrama por litro.",
    "O sistema de aquisição registrou transientes eletromagnéticos durante a descarga atmosférica.",
    "A análise de componentes principais reduziu a dimensionalidade preservando noventa por cento da variância.",
    "O cristão líquido apresentou transição de fase sob resfriamento controlado.",
    "A turbina operou em regime permanente com rendimento hidráulico acima de noventa por cento.",
    "O ensaio de fadiga interrompeu-se após um milhão de ciclos sem propagação de trinca.",
    "A difração de raios X identificou a fase cristalina majoritária como quartzo.",
    "O controlador proporcional integral derivado estabilizou a malha de temperatura em dois segundos.",
    "A cromatografia líquida separou os analitos em cinco picos bem resolvidos.",
    "O modelo de regressão obteve coeficiente de determinação igual a zero vírgula noventa e três.",
    "A antena transmissoira operou na faixa de dois vírgula quatro gigahertz com ganho de três decibéis.",
    "O ensaio de tração seguiu o procedimento padrão com velocidade constante de deformação.",
    "A estimativa de parâmetros usou o método da máxima verossimilhança com inicialização aleatória.",
    "O trocador de calor atingiu eficiência global de setenta e oito por cento no regime nominal.",
    "A imagem de ressonância magnética evidenciou contraste entre tecidos moles sem meio de contraste.",
    "O oscilador de referência apresentou deriva de frequência dentro da tolerância especificada.",
    "A destilação fracionada separou os componentes com pureza superior a noventa e nove por cento.",
    "O experimento de dupla fenda confirmou o padrão de interferência previsto pela teoria ondulatória.",
    "A filtragem passa baixas atenuou o ruído de alta frequência sem distorcer o sinal útil.",
    "O dimensionamento estrutural considerou coeficientes de majoração de ações e minoração de resistências.",
]


def main() -> int:
    rnd = random.Random(2026)
    paragrafos = []
    for _ in range(220):
        k = rnd.randint(5, 9)
        inicio = rnd.randrange(len(FRASES))
        ordem = [FRASES[(inicio + i) % len(FRASES)] for i in range(k)]
        paragrafos.append(" ".join(ordem))
    DESTINO.write_text("\n\n".join(paragrafos) + "\n", encoding="utf-8")
    print(f"[ok] {DESTINO} ({DESTINO.stat().st_size:,} bytes, {len(paragrafos)} parágrafos)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
